"""模型预测程序"""
from home import *
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib
matplotlib.use('Qt5Agg')  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanves
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from rle import rle_decode
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torch

def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script


from rle import rle_encode
import argparse
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import torchvision
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyFigure(FigureCanves):
    # def __init__(self, width, height, dpi):
    def __init__(self):
        # self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.fig = plt.figure()
        super(MyFigure, self).__init__(self.fig)  # 在父类中激活Figure窗口


class MainWindow(QMainWindow, Ui_MainWindow):
    _startPos = None
    _endPos = None
    _isTracking = False
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        """相关组件"""
        # 按钮
        self.but_AddModelPath = self.but_addmodel
        self.but_AddModelPath.clicked.connect(self.clcikLoadModelPath)
        self.but_AddImgPath = self.But_loaddata
        self.but_AddImgPath.clicked.connect(self.clickLoadDatasetFliePath)
        self.but_ModelPredict = self.But_predit
        self.but_ModelPredict.clicked.connect(self.model_predict)
        self.but_LoadTmpPath = self.But_readtmp
        self.but_LoadTmpPath.clicked.connect(self.clickLoadImgPath)
        self.but_ShowImg = self.But_showimg
        self.but_ShowImg.clicked.connect(self.showImg)
        self.but_LastImg = self.last_img
        self.but_LastImg.clicked.connect(self.lastImg)
        self.but_NextImg = self.next_img
        self.but_NextImg.clicked.connect(self.nextImg)
        # 地址显示文本框
        self.LineEdit_ModelPath = self.model_path
        self.LineEdit_ImgPath = self.datesets_path
        self.LineEdit_TmpPath = self.tmp_pth
        # 输入文本框
        self.LineEdit_Idx = self.Entry_idx
        # 进度条
        self.ProgressBar_Predict = self.progressBar
        # 模型多按钮
        self.ComboBox_ChoseModel = self.comboBox
        self.ComboBox_ChoseModel.currentIndexChanged.connect(self.clickComboBoxItem)
        # 显示图片框
        self.ShowOriginalImg = QGridLayout(self.groupBox_2)
        self.ShowProcessImg = QGridLayout(self.groupBox_3)
        self.ShowOriginalImg.setContentsMargins(0, 0, 0, 0)
        self.ShowProcessImg.setContentsMargins(0, 0, 0, 0)

        self.id = 0
        self.count = 0  # 绘图标记

        # 弹窗
        info = ('------------------------------------------------------------------\n'
                '！！请按照正确方式使用程序！！\n'
                '！！此程序目前只适用于天池比赛：地表建筑物识别！！\n'
                '使用注意事项如下\n'
                '需要有使用pytorch预训练的模型（.pth文件）\n'
                '模型预测时程序会出现假死现象，不要强制退出，程序还在正常运行\n'
                '模型预测时需要注意以下事项：\n'
                '1.把比赛下载的数据集放到和程序同一目录的datasets文件夹下\n'
                'test_a_samplesubmit.csv的路径为：./datasets/test_a_samplesubmit.csv\n'
                'test_a文件夹的路径为：./datasets/test_a\n'
                '程序调用时使用的相对路径\n'
                '2.模型文件.pth的命名格式如下：\n'
                'fcn_resnet50_15_model.pth\n'
                '   fcn表示模型\n'
                '   rennet50表示主干网络\n'
                '  15表示训练次数\n'
                '  注:_分割，前俩个命名方式必须按照格式命名，!!全部小写!!\n'
                '3.目前只使用以下模型\n'
                'FCN+ResNet50\n'
                'FCN+ResNet101\n'
                'Unet+ResNet50\n'
                'Unet+EfficientB4\n'
                'UnetPlusPlus+EfficientB4\n'
                '如需增加模型要改predictor.py的代码\n'
                '------------------------------------------------------------------\n')
        about = QMessageBox()
        about.setText(info)
        about.exec()

    # 重写移动事件
    '''
    def mouseMoveEvent(self, e: QMouseEvent):
        self._endPos = e.pos() - self._startPos
        self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._isTracking = True
            self._startPos = QPoint(e.x(), e.y())
    '''

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    # 读取训练完成的文件 按下
    def clickLoadImgPath(self):
        tmp_path_tuple = QtWidgets.QFileDialog.getOpenFileName(None, '选择csv文件路径', filter='CSV File (*.csv)')
        tmp_path = tmp_path_tuple[0]
        self.LineEdit_TmpPath.setText(tmp_path)
        return tmp_path

    # 添加模型按钮 按下
    def clcikLoadModelPath(self):
        model_path_tuple = QtWidgets.QFileDialog.getOpenFileNames(None, '选择与训练模型文件路径', filter='model(.pth) File (*.pth)')
        model_path_list = model_path_tuple[0]
        model_path = ''
        for i in range(len(model_path_list)):
            self.ComboBox_ChoseModel.addItem(model_path_list[i])
            model_path = model_path + model_path_list[i] + '；'
        self.LineEdit_ModelPath.setText(model_path)

    # 航拍地标建筑物数据集按钮 按下
    def clickLoadDatasetFliePath(self):
        dataset_FilePath = QtWidgets.QFileDialog.getExistingDirectory(None, '选择数据集文件夹')
        self.LineEdit_ImgPath.setText(dataset_FilePath)

    # 选择模型多选框 选择后的函数
    def clickComboBoxItem(self):
        model_path = self.ComboBox_ChoseModel.currentText()
        '''
        # 输出选项集合中每个选项的索引与对应的内容
        # count()：返回选项集合中的数目
        for count in range(self.ComboBox_ChoseModel.count()):
            print(self.ComboBox_ChoseModel.currentText())
            print(self.ComboBox_ChoseModel.itemText(count))
        '''
        return model_path

    # 加载图片
    def loadImg(self):
        # 加载数据集
        tmp_path = self.LineEdit_TmpPath.text()
        datasets_path = self.LineEdit_ImgPath.text()
        result = pd.read_csv(tmp_path, sep='\t', names=['name', 'mask'])
        img = cv2.imdecode(np.fromfile(file=(datasets_path + '/' + result['name'].iloc[self.id]), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if result['mask'].iloc[self.id] is np.NAN:
            mask = cv2.imdecode(np.fromfile(file=(datasets_path + '/' + result['name'].iloc[self.id]), dtype=np.uint8),
                                cv2.IMREAD_COLOR)
        else:
            mask = rle_decode(result['mask'].iloc[self.id])
        return img, mask

    # 查看图片按钮 按下
    def showImg(self):
        tmp_path = self.LineEdit_TmpPath.text()
        datasets_path = self.LineEdit_ImgPath.text()
        if tmp_path == '':
            self.clickLoadImgPath()
        if datasets_path == '':
            self.clickLoadDatasetFliePath()
        else:

            if self.LineEdit_Idx.text() != "":
                self.id = int(self.LineEdit_Idx.text())
                if 0 < self.id >= 2499:
                    msg_box = QMessageBox(QMessageBox.Warning, 'Warning haha!!!!', "输入正确的图像序号，不然只显示第一张图片")
                    msg_box.exec_()
                    self.id = 0
            if self.LineEdit_Idx.text() == '':
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning haha!!!!', "请输入图片序号， 0-2500之间")
                msg_box.exec_()

            # 加载数据集
            img, mask = self.loadImg()
            # 绘制图像
            if self.count == 0:
                self.oF = MyFigure()
                plt.imshow(img)
                plt.axis("off")
                self.ShowOriginalImg.addWidget(self.oF)
                self.pF = MyFigure()
                plt.imshow(mask)
                plt.axis("off")
                self.ShowProcessImg.addWidget(self.pF)
                self.count += 1
            else:
                self.ShowProcessImg.removeWidget(self.oF)
                self.oF.deleteLater()
                self.ShowProcessImg.removeWidget(self.pF)
                self.pF.deleteLater()
                self.oF = MyFigure()
                plt.imshow(img)
                plt.axis("off")
                self.ShowOriginalImg.addWidget(self.oF)
                self.pF = MyFigure()
                plt.imshow(mask)
                plt.axis("off")
                self.ShowProcessImg.addWidget(self.pF)

    # 上一张按钮 按下
    def lastImg(self):
        if self.LineEdit_Idx.text() != "":
            if int(self.LineEdit_Idx.text()) <= 0:
                self.id = 0
            elif int(self.LineEdit_Idx.text()) > 2500:
                self.id = 2500
            else:
                self.id = int(self.LineEdit_Idx.text()) - 1
                self.LineEdit_Idx.setText(str(self.id))
            img, mask = self.loadImg()
            self.ShowProcessImg.removeWidget(self.oF)
            self.oF.deleteLater()
            self.ShowProcessImg.removeWidget(self.pF)
            self.pF.deleteLater()
            self.oF = MyFigure()
            plt.imshow(img)
            plt.axis("off")
            self.ShowOriginalImg.addWidget(self.oF)
            self.pF = MyFigure()
            plt.imshow(mask)
            plt.axis("off")
            self.ShowProcessImg.addWidget(self.pF)

    # 下一张按钮 按下
    def nextImg(self):
        if self.LineEdit_Idx.text() != "":
            if int(self.LineEdit_Idx.text()) < 0:
                self.id = 0
            elif int(self.LineEdit_Idx.text()) >= 2499:
                self.id = 2499
            else:
                self.id = int(self.LineEdit_Idx.text()) + 1
                self.LineEdit_Idx.setText(str(self.id))
            img, mask = self.loadImg()
            self.ShowProcessImg.removeWidget(self.oF)
            self.oF.deleteLater()
            self.ShowProcessImg.removeWidget(self.pF)
            self.pF.deleteLater()
            self.oF = MyFigure()
            plt.imshow(img)
            plt.axis("off")
            self.ShowOriginalImg.addWidget(self.oF)
            self.pF = MyFigure()
            plt.imshow(mask)
            plt.axis("off")
            self.ShowProcessImg.addWidget(self.pF)

    def parse_args(self):
        model_path = self.clickComboBoxItem()
        dataset_path = self.LineEdit_ImgPath.text() + '/'
        parser = argparse.ArgumentParser(description='Train semantic segmentation network')
        parser.add_argument('--modelDir',
                            help='saved model path name',
                            default=model_path,
                            type=str)
        parser.add_argument('--data_path',
                            help='dataset path',
                            default=dataset_path,
                            type=str)
        parser.add_argument('--image_size',
                            help='total train epoch num',
                            default=256,
                            type=int)
        parser.add_argument('--gpu_ids',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU',
                            default=[0],
                            type=str)
        args = parser.parse_args()
        return args

    # 模型预测按钮 按下
    def model_predict(self):
        model_path = self.clickComboBoxItem()
        path = self.LineEdit_ModelPath.text()
        dataset_path = self.LineEdit_ImgPath.text()
        if dataset_path == '':
            self.clickLoadDatasetFliePath()
        if path == "":
            self.clcikLoadModelPath()
        if model_path != '选择模型':
            args = self.parse_args()
            model_fliename_ls = model_path.split('/')[-1]
            model_name_ls = model_fliename_ls.split('_')
            model_name = model_name_ls[0] + '_' + model_name_ls[1]
            trfm = T.Compose([
                T.ToPILImage(),
                T.Resize(args.image_size),
                T.ToTensor(),
                T.Normalize([0.625, 0.448, 0.688],
                            [0.131, 0.177, 0.101]),
            ])
            subm = []
            if model_name == 'unet_renet50':
                model = smp.Unet(
                    encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=1,  # model output channels (number of classes in your dataset)
                )
            elif model_name == 'unet_efficientB4':
                model = smp.Unet(
                    encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=1,  # model output channels (number of classes in your dataset)
                )
            elif model_name == 'unet++_efficientB4':
                model = smp.UnetPlusPlus(
                    encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
                    in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                    classes=1,  # model output channels (number of classes in your dataset)
                )
            elif model_name == 'fcn_resnet101':
                model = torchvision.models.segmentation.fcn_resnet101(True)
                model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            elif model_name == 'fcn_resnet50':
                model = torchvision.models.segmentation.fcn_resnet50(True)
                model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            model.to(DEVICE)
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids, output_device=0)
            if os.path.exists(args.modelDir):
                checkpoint = torch.load(args.modelDir)
                model.load_state_dict(checkpoint['state_dict'])
                print("load model from {}".format(args.modelDir))
            model.eval()
            test_mask = pd.read_csv('./datasets/test_a_samplesubmit.csv', sep='\t',
                                    names=['name', 'mask'])
            test_mask['name'] = test_mask['name'].apply(lambda x: os.path.join(args.data_path) + x)

            for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
                self.progressBar.setRange(0, 2499)
                self.progressBar.setValue(idx)
                # image = cv2.imread(name)
                result = pd.read_csv('./datasets/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
                image = cv2.imdecode(
                    np.fromfile(file=(dataset_path + '/' + result['name'].iloc[idx]), dtype=np.uint8),
                    cv2.IMREAD_COLOR)
                image = trfm(image)
                with torch.no_grad():
                    image = image.to(DEVICE)[None]
                    if model_name_ls[0] == 'fcn':
                        score = model(image)['out'][0][0]
                    elif model_name_ls[0] == 'unet' or 'unet++':
                        score = model(image)[0][0]
                    score_sigmoid = score.sigmoid().cpu().numpy()
                    score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
                    score_sigmoid = cv2.resize(score_sigmoid, (512, 512))
                    # break
                subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
            subm = pd.DataFrame(subm)
            subm.to_csv('./tmp.csv', index=None, header=None, sep='\t')
        else:
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning haha!!!!', "请选择模型")
            msg_box.exec_()


def run():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run()