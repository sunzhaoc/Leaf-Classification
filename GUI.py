
from PyQt5.QtWidgets import (QLineEdit, QFileDialog, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QWidget)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import sys
import final


def switch_case(value):
    switcher = {
        12: "冬青",
        13: "樟树",
        14: "女贞",
        15: "金边女贞",
        16: "万寿菊",
        17: "红花檵木",
        18: "枸骨",
        19: "花叶青木",
        20: "石楠",
        21: "八角金盘",
        22: "侧柏",
        23: "金边黄杨",
    }
    return switcher.get(value, None)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.ansPredict_1 = 100
        self.ansPredict_2 = 0
        self.ansPredict_3 = 0
        self.myFileStr = './leaf_test/00018/18204.jpg'  # 这是inference的输入
        self.ans = '植物'  # 这是inference的结果
        self.LineEdit = QLineEdit(self.myFileStr)

        # ok按钮
        self.okButton = QPushButton("点击识别")
        self.okButton.clicked.connect(self.showAns)

        # 浏览目录按钮
        self.openButton = QPushButton("...")
        self.openButton.clicked.connect(self.showDialog)
        self.openButton.setFixedSize(23, 23)

        pixmap = QPixmap(self.myFileStr,'wr')

        # 原始图片
        self.graphLabel = QLabel()
        self.graphLabel.setPixmap(pixmap)
        self.graphLabel.setScaledContents(True)
        self.graphLabel.setFixedSize(200, 200)   # 长， 高
        self.graphLabel.setAlignment(Qt.AlignCenter)
        self.graphText = QLabel()
        self.graphText.setText('待识别图片：')
        self.graphText.setFont(QFont("ZhongSong", 20, QFont.Bold))

        # 识别结果1
        self.ansText_11 = QLabel()
        self.ansText_11.setText("识别结果1：")
        self.ansText_11.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.ansText_12 = QLabel()
        self.ansText_12.setText("可能性：{0}%".format(self.ansPredict_1))
        self.ansText_12.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.vbox_1 = QVBoxLayout()
        self.vbox_1.addWidget(self.ansText_11)
        self.vbox_1.addWidget(self.ansText_12)

        # 识别结果2
        self.ansText_21 = QLabel()
        self.ansText_21.setText("识别结果2：")
        self.ansText_21.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.ansText_22 = QLabel()
        self.ansPredict_1 = 100
        self.ansText_22.setText("可能性：{0}%".format(self.ansPredict_2))
        self.ansText_22.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.vbox_2 = QVBoxLayout()
        self.vbox_2.addWidget(self.ansText_21)
        self.vbox_2.addWidget(self.ansText_22)

        # 识别结果3
        self.ansText_31 = QLabel()
        self.ansText_31.setText("识别结果3：")
        self.ansText_31.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.ansText_32 = QLabel()
        self.ansPredict_1 = 100
        self.ansText_32.setText("可能性：{0}%".format(self.ansPredict_3))
        self.ansText_32.setFont(QFont("ZhongSong", 20, QFont.Bold))
        self.vbox_3 = QVBoxLayout()
        self.vbox_3.addWidget(self.ansText_31)
        self.vbox_3.addWidget(self.ansText_32)

        hbox_1 = QHBoxLayout()
        hbox_1.addWidget(self.LineEdit)
        hbox_1.addWidget(self.openButton)
        hbox_1.addWidget(self.okButton)

        hbox_2 = QHBoxLayout()
        hbox_2.addWidget(self.graphText)
        hbox_2.addWidget(self.graphLabel)

        hbox_3 = QHBoxLayout()
        hbox_3.addLayout(self.vbox_1)

        hbox_4 = QHBoxLayout()
        hbox_4.addLayout(self.vbox_2)

        hbox_5 = QHBoxLayout()
        hbox_5.addLayout(self.vbox_3)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_1)
        vbox.addSpacing(30)
        vbox.addLayout(hbox_2)
        vbox.addLayout(hbox_3)
        vbox.addLayout(hbox_4)
        vbox.addLayout(hbox_5)

        self.setLayout(vbox)

        self.setGeometry(200,200,294,1100)
        self.setWindowTitle("植物图像识别工具")
        self.show()

    def showDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', './leaf_test/')

        if fname[0]:
            f = open(fname[0], 'r')
            self.myFileStr = fname[0]
            self.LineEdit.setText(fname[0])
            pixmap = QPixmap(self.myFileStr)
            self.graphLabel.setPixmap(pixmap)
            print(self.myFileStr)

    def showAns(self):
        final_predict_val, final_predict_index = final.inference(self.myFileStr)
        ansIndex_1=final_predict_index[0][0]
        ansIndex_2=final_predict_index[0][1]
        ansIndex_3=final_predict_index[0][2]

        self.ansPredict_1 = final_predict_val[0][0]*100
        self.ansPredict_2 = final_predict_val[0][1]*100
        self.ansPredict_3 = final_predict_val[0][2]*100
        self.ansText_11.setText("识别结果1：{0}".format(switch_case(ansIndex_1)))
        self.ansText_21.setText("识别结果2：{0}".format(switch_case(ansIndex_2)))
        self.ansText_31.setText("识别结果3：{0}".format(switch_case(ansIndex_3)))

        self.ansText_12.setText("可能性：{:.2f}%".format(self.ansPredict_1))
        self.ansText_22.setText("可能性：{:.2f}%".format(self.ansPredict_2))
        self.ansText_32.setText("可能性：{:.2f}%".format(self.ansPredict_3))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MyWindow()
    sys.exit(app.exec_())