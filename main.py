from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from collections import defaultdict
from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
import numpy as np
import time
import json
import torch
import sys
import cv2
import os


# make this suitable for video and image
class BasePredictor(QObject):
    main_pre_img = Signal(np.ndarray)  # raw image signal
    main_res_img = Signal(np.ndarray)  # test result signal
    main_status_msg = Signal(
        str
    )  # Detecting/pausing/stopping/testing complete/error reporting signal
    # main_fps = Signal(str)  # fps
    # main_labels = Signal(dict)  # Detected target results (number of each category)
    # main_progress = Signal(int)  # Completeness

    def __init__(self, model):
        QObject.__init__(self)
        # GUI args
        self.used_model_name = None  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = ""  # input source
        self.stop_dtc = False  # Terminate
        self.continue_dtc = True  # pause
        self.save_res = False  # Save test results
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar

        # Usable if setup is done
        # model.eval()
        self.model = model
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.data_path = None
        self.source_type = None
        self.batch = None
        # self.warmup()

    def warmup(self):
        raise NotImplementedError

    def preprocess(self, img):
        raise NotImplementedError

    def postprocess(self, img):
        raise NotImplementedError

    def write_results(self, result):
        raise NotImplementedError

    @torch.no_grad()
    def run(self):
        raise NotImplementedError


class VirtualClothingPredictor(BasePredictor):
    time_msg = Signal(str)

    def __init__(self):
        super().__init__(None)
        self.used_model_name = ""  # The detection model name to use
        self.new_model_name = ""  # Models that change in real time
        self.source = ""  # input source
        self.cloth = ""
        self.save_res = False  # Save test results
        self.save_dir = ""
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results

        # Usable if setup is done
        self.imgsz = (512, 512)
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_fp16 = False
        self.pre_time = None
        self.infer_time = None
        self.post_time = None

    def warmup(self):
        image = torch.randn(1, 3, *self.imgsz)
        for _ in range(3):
            self.model(image)

    def preprocess(self):
        st = time.time()
        img = cv2.imread(self.source)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.is_fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        self.pre_time = time.time() - st
        return img

    def postprocess(self, img):
        return img

    def write_results(self, result):
        if torch.cuda.is_available():
            result = result.cpu()
        img_np = result.numpy()
        file_name = self.source.split(os.sep)[-1]
        save_path = os.path.join(self.save_dir, file_name)
        cv2.imwrite(save_path, img_np)

    @torch.no_grad()
    def run(self):
        if not self.save_dir:
            self.save_dir = os.sep.join(self.source.split(os.sep)[:-1])
        os.makedirs(self.save_dir, exist_ok=True)

        img = self.preprocess()

        st = time.time()
        out = self.model(img)
        self.infer_time = time.time() - st

        if self.save_res:
            self.write_results(out)


class MainWindow(QMainWindow, Ui_MainWindow):
    begin_sgl = (
        Signal()
    )  # The main window sends an execution signal to the yolo instance

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # basic interface
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # rounded transparent
        self.setWindowFlags(
            Qt.FramelessWindowHint
        )  # Set window flag: hide window borders
        UIFuncitons.uiDefinitions(self)
        # Show module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # read model folder
        self.pt_list = os.listdir("./models")
        self.pt_list = [file for file in self.pt_list if file.endswith(".pt")]
        self.pt_list.sort(
            key=lambda x: os.path.getsize("./models/" + x)
        )  # sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        # self.Qtimer_ModelBox = QTimer(
        #     self
        # )  # Timer: Monitor model file changes every 2 seconds
        # self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        # self.Qtimer_ModelBox.start(2000)

        # thread
        self.predictor = VirtualClothingPredictor()  # Create a Yolo instance
        self.select_model = self.model_box.currentText()  # default model
        self.predictor.new_model_name = "./models/%s" % self.select_model
        self.thread = QThread()  # Create yolo thread
        self.predictor.main_pre_img.connect(
            lambda x: self.show_image(x, self.pre_video)
        )
        self.predictor.main_res_img.connect(
            lambda x: self.show_image(x, self.res_video)
        )
        self.predictor.main_status_msg.connect(lambda x: self.show_status(x))
        # self.predictor.main_fps.connect(lambda x: self.fps_label.setText(x))
        # self.yolo_predict.yolo2main_labels.connect(self.show_labels)
        # self.predictor.main_class_num.connect(
        #     lambda x: self.Class_num.setText(str(x))
        # )
        # self.predictor.main_target_num.connect(
        #     lambda x: self.Target_num.setText(str(x))
        # )
        # self.predictor.main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.begin_sgl.connect(self.predictor.run)
        self.predictor.moveToThread(self.thread)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        # self.iou_spinbox.valueChanged.connect(
        #     lambda x: self.change_val(x, "iou_spinbox")
        # )  # iou box
        # self.iou_slider.valueChanged.connect(
        #     lambda x: self.change_val(x, "iou_slider")
        # )  # iou scroll bar
        # self.conf_spinbox.valueChanged.connect(
        #     lambda x: self.change_val(x, "conf_spinbox")
        # )  # conf box
        # self.conf_slider.valueChanged.connect(
        #     lambda x: self.change_val(x, "conf_slider")
        # )  # conf scroll bar
        # self.speed_spinbox.valueChanged.connect(
        #     lambda x: self.change_val(x, "speed_spinbox")
        # )  # speed box
        # self.speed_slider.valueChanged.connect(
        #     lambda x: self.change_val(x, "speed_slider")
        # )  # speed scroll bar

        # Prompt window initialization
        # self.Class_num.setText("--")
        # self.Target_num.setText("--")
        # self.fps_label.setText("--")
        self.Model_name.setText(self.select_model)

        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # select local file
        # self.src_cam_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_cam
        # self.src_rtsp_button.clicked.connect(self.show_status("The function has not yet been implemented."))#chose_rtsp

        # start testing button
        self.run_button.clicked.connect(self.run_or_continue)  # pause/start
        self.stop_button.clicked.connect(self.stop)  # termination

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # save image option
        # self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        self.ToggleBotton.clicked.connect(
            lambda: UIFuncitons.toggleMenu(self, True)
        )  # left navigation button
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  # top right settings button

        # initialization
        self.load_config()

    # The main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep the original data ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[2] * frame.shape[1],
                QImage.Format_RGB888,
            )
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # Control start/pause
    def run_or_continue(self):
        if self.predictor.source == "":
            self.show_status(
                "Please select a video source before starting detection..."
            )
            self.run_button.setChecked(False)
        else:
            self.predictor.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # start button
                self.save_txt_button.setEnabled(
                    False
                )  # It is forbidden to check and save after starting the detection
                self.save_res_button.setEnabled(False)
                self.show_status("Detecting...")
                self.predictor.continue_dtc = True  # Control whether Yolo is paused
                if not self.thread.isRunning():
                    self.thread.start()
                    self.begin_sgl.emit()

            else:
                self.predictor.continue_dtc = False
                self.show_status("Pause...")
                self.run_button.setChecked(False)  # start button

    # bottom status bar information
    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == "Detection completed" or msg == "检测完成":
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.thread.isRunning():
                self.thread.quit()  # end process
        elif msg == "Detection terminated!" or msg == "检测终止":
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.thread.isRunning():
                self.thread.quit()  # end process
            self.pre_video.clear()  # clear image display
            self.res_video.clear()
            self.Class_num.setText("--")
            self.Target_num.setText("--")
            self.fps_label.setText("--")

    # select local file
    def open_src_file(self):
        config_file = "config/fold.json"
        config = json.load(open(config_file, "r", encoding="utf-8"))
        open_fold = config["open_fold"]
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(
            self,
            "image",
            open_fold,
            "Pic File(*.jpg *.png)",
        )
        if name:
            self.predictor.source = name
            self.show_status("Load File：{}".format(os.path.basename(name)))
            config["open_fold"] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(config_json)
            name, _ = QFileDialog.getOpenFileName(
                self,
                "image",
                open_fold,
                "Pic File(*.jpg *.png)",
            )
            if name:
                self.predictor.cloth = name
            else:
                self.predictor.source = ""
            self.stop()

    # Select camera source----  have one bug
    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.close_button,
                title="Note",
                text="loading camera...",
                time=2000,
                auto=True,
            ).exec()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet(
                """
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            """
            )

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).x()
            y = self.src_cam_button.mapToGlobal(self.src_cam_button.pos()).y()
            y = y + self.src_cam_button.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec(pos)
            if action:
                self.predictor.source = action.text()
                self.show_status("Loading camera：{}".format(action.text()))

        except Exception as e:
            self.show_status("%s" % e)

    # select network source
    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = "config/ip.json"
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.2:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, "r", encoding="utf-8"))
            ip = config["ip"]
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(
            lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text())
        )

    # load network sources
    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.close_button, title="提示", text="加载 rtsp...", time=1000, auto=True
            ).exec()
            self.predictor.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open("config/ip.json", "w", encoding="utf-8") as f:
                f.write(new_json)
            self.show_status("Loading rtsp：{}".format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.show_status("%s" % e)

    # Save test result button--picture/video
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status("NOTE: Run image results are not saved.")
            self.predictor.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status("NOTE: Run image results will be saved.")
            self.predictor.save_res = True

    # Save test result button -- label (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status("NOTE: Labels results are not saved.")
            self.predictor.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status("NOTE: Labels results will be saved.")
            self.predictor.save_txt = True

    # Configuration initialization  ~~~wait to change~~~
    def load_config(self):
        config_file = "config/setting.json"
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {
                "iou": iou,
                "conf": conf,
                "rate": rate,
                "save_res": save_res,
                "save_txt": save_txt,
            }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, "r", encoding="utf-8"))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config["iou"]
                conf = config["conf"]
                rate = config["rate"]
                save_res = config["save_res"]
                save_txt = config["save_txt"]
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.predictor.save_res = False if save_res == 0 else True
        # self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        # self.predictor.save_txt = False if save_txt == 0 else True
        self.run_button.setChecked(False)
        self.show_status("Welcome~")

    # Terminate button and associated state
    def stop(self):
        if self.thread.isRunning():
            self.thread.quit()  # end thread
        self.predictor.stop_dtc = True
        self.run_button.setChecked(False)  # start key recovery
        self.save_res_button.setEnabled(True)  # Ability to use the save button
        # self.save_txt_button.setEnabled(True)  # Ability to use the save button
        self.pre_video.clear()  # clear image display
        self.res_video.clear()  # clear image display
        # self.progress_bar.setValue(0)
        self.Class_num.setText("--")
        self.Target_num.setText("--")
        self.fps_label.setText("--")

    # Change detection parameters
    # def change_val(self, x, flag):
        # if flag == "iou_spinbox":
        #     self.iou_slider.setValue(
        #         int(x * 100)
        #     )  # The box value changes, changing the slider
        # elif flag == "iou_slider":
        #     self.iou_spinbox.setValue(
        #         x / 100
        #     )  # The slider value changes, changing the box
        #     self.show_status("IOU Threshold: %s" % str(x / 100))
        #     self.predictor.iou_thres = x / 100
        # elif flag == "conf_spinbox":
        #     self.conf_slider.setValue(int(x * 100))
        # elif flag == "conf_slider":
        #     self.conf_spinbox.setValue(x / 100)
        #     self.show_status("Conf Threshold: %s" % str(x / 100))
        #     self.predictor.conf_thres = x / 100
        # elif flag == "speed_spinbox":
        #     self.speed_slider.setValue(x)
        # elif flag == "speed_slider":
        #     self.speed_spinbox.setValue(x)
        #     self.show_status("Delay: %s ms" % str(x))
        #     self.predictor.speed_thres = x  # ms

    # change model
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        self.predictor.new_model_name = "./models/%s" % self.select_model
        self.show_status("Change Model：%s" % self.select_model)
        self.Model_name.setText(self.select_model)

    # Cycle monitoring model file changes
    def ModelBoxRefre(self):
        pt_list = os.listdir("./models")
        pt_list = [file for file in pt_list if file.endswith(".pt")]
        pt_list.sort(key=lambda x: os.path.getsize("./models/" + x))
        # It must be sorted before comparing, otherwise the list will be refreshed all the time
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # Get the mouse position (used to hold down the title bar and drag the window)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize the adjustment when dragging the bottom and right edges of the window size
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # Exit Exit thread, save settings
    def closeEvent(self, event):
        config_file = "config/setting.json"
        config = dict()
        config["iou"] = self.iou_spinbox.value()
        config["conf"] = self.conf_spinbox.value()
        config["rate"] = self.speed_spinbox.value()
        config["save_res"] = (
            0 if self.save_res_button.checkState() == Qt.Unchecked else 2
        )
        config["save_txt"] = (
            0 if self.save_txt_button.checkState() == Qt.Unchecked else 2
        )
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_json)
        # Exit the process before closing
        if self.thread.isRunning():
            self.predictor.stop_dtc = True
            self.thread.quit()
            MessageBox(
                self.close_button,
                title="Note",
                text="Exiting, please wait...",
                time=3000,
                auto=True,
            ).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
