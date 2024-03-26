import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QRadioButton, QTextEdit
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPixmap
import os

class ImageFinderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.Damping_Ratio = 100
        self.initUI()
    
    def initUI(self):

        self.setWindowTitle('Pulse-Speed-Torque Data for Motor Control System')
        self.setGeometry(100, 100, 1200, 900)
    
        # self.centralwidget = QWidget(self)
        # self.centralwidget.setObjectName(u"centralwidget")
        # self.widget = PlotWidget(self.centralwidget)
        # self.widget.setObjectName(u"widget")
        # self.widget.setGeometry(QRect(20, 130, 301, 411))
        # self.widget_2 = PlotWidget(self.widget)
        # self.widget_2.setObjectName(u"widget_2")
        # self.widget_2.setGeometry(QRect(320, 0, 301, 411))

        self.horizontallabel_0 = QLabel(self)
        #self.horizontalLayout_0 = QHBoxLayout(self)
        self.horizontallabel_0.setObjectName(u"Motor_Parameter_Values")
        self.horizontallabel_0.setGeometry(QRect(0, 100, 1200, 101))
        # set style of QHboxLayout
        self.horizontallabel_0.setStyleSheet("QLabel"
                                 "{" 
                                 "border : 4px solid black;"
                                 "background : white;"
                                 "}")                         
        self.textIR = QLabel(self.horizontallabel_0)
        self.textIR.setObjectName(u"Inertia_Ratio")
        self.textIR.setGeometry(QRect(0, 40, 100, 51))
        self.textIR.setText("Inertia_Ratio[%]")
        # set style of QLabel to default
        self.textIR.setStyleSheet("QLabel")
        #self.textIR.setStyleSheet("QLabel")

        self.textEdit_IR = QTextEdit(self.horizontallabel_0)
        self.textEdit_IR.setObjectName(u"Inertia_Ratio")
        self.textEdit_IR.setGeometry(QRect(100, 40, 104, 51))
        self.textEdit_IR.setText("Inertia_Ratio")
        self.textEdit_FP = QTextEdit(self.horizontallabel_0)
        self.textEdit_FP.setObjectName(u"First_Pole")
        self.textEdit_FP.setGeometry(QRect(250, 40, 104, 51))
        self.textEdit_FP.setText("First_Pole")
        self.textEdit_SP = QTextEdit(self.horizontallabel_0)
        self.textEdit_SP.setObjectName(u"Second_Pole")
        self.textEdit_SP.setText("Second_Pole")
        self.textEdit_SP.setGeometry(QRect(490, 40, 104, 51))
        self.textEdit_IP = QTextEdit(self.horizontallabel_0)
        self.textEdit_IP.setObjectName(u"Integral_Pole")
        self.textEdit_IP.setText("Integral_Pole")
        self.textEdit_IP.setGeometry(QRect(650, 40, 104, 51))


        self.horizontallabel_1 = QLabel(self)#QWidget(self)
        self.horizontallabel_1.setObjectName(u"Motor_Type_Widget")
        
        self.horizontallabel_1.setText("(Left)Motor Type")
        self.horizontallabel_1.setGeometry(QRect(0, 200, 600, 101))
        self.horizontallabel_1.setStyleSheet("QLabel"
                                 "{"
                                 "border : 4px solid black;"
                                 "background : white;"
                                 "}")

        self.horizontalLayout_1 = QHBoxLayout(self.horizontallabel_1)
        self.horizontalLayout_1.setObjectName(u"horizontalLayout")        
        
        self.radioButton_1 = QRadioButton(self.horizontallabel_1)
        self.radioButton_1.setObjectName(u"Hyeon_Seo")
        self.horizontalLayout_1.addWidget(self.radioButton_1)
        self.radioButton_1.setText("Hyeon_Seo")   

        self.radioButton_2 = QRadioButton(self.horizontallabel_1)
        self.radioButton_2.setObjectName(u"Small_Joint")
        self.horizontalLayout_1.addWidget(self.radioButton_2)
        self.radioButton_2.setText("Small_Joint")

        self.radioButton_3 = QRadioButton(self.horizontallabel_1)
        self.radioButton_3.setObjectName(u"Zero_460W")
        self.horizontalLayout_1.addWidget(self.radioButton_3)
        self.radioButton_3.setText("Zero_460W")

        self.horizontallabel_2 = QLabel(self)#QWidget(self)
        self.horizontallabel_2.setObjectName(u"Motor_Type_Widget")
        self.horizontallabel_2.setText("(Right)Motor Type")
        self.horizontallabel_2.setGeometry(QRect(620, 200, 600, 101))
        self.horizontallabel_2.setStyleSheet("QLabel"
                                 "{"
                                 "border : 4px solid black;"
                                 "background : white;"
                                 "}")

        self.horizontalLayout_2 = QHBoxLayout(self.horizontallabel_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout")        
        
        self.radioButton_4 = QRadioButton(self.horizontallabel_2)
        self.radioButton_4.setObjectName(u"Hyeon_Seo")
        self.horizontalLayout_2.addWidget(self.radioButton_4)
        self.radioButton_4.setText("Hyeon_Seo")   

        self.radioButton_5 = QRadioButton(self.horizontallabel_2)
        self.radioButton_5.setObjectName(u"Small_Joint")
        self.horizontalLayout_2.addWidget(self.radioButton_5)
        self.radioButton_5.setText("Small_Joint")

        self.radioButton_6 = QRadioButton(self.horizontallabel_2)
        self.radioButton_6.setObjectName(u"Zero_460W")
        self.horizontalLayout_2.addWidget(self.radioButton_6)
        self.radioButton_6.setText("Zero_460W")

        '''
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(40, 20, 361, 21))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lineEdit = QLineEdit(self.verticalLayoutWidget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.verticalLayout.addWidget(self.lineEdit)

        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(20, 0, 104, 21))
        self.textEdit_2 = QTextEdit(self.centralwidget)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(20, 40, 104, 21))
        self.horizontalLayoutWidget_2 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(190, 60, 111, 80))
        self.horizontalLayout_2 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayoutWidget_3 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(20, 60, 111, 80))
        self.horizontalLayout_3 = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayoutWidget_4 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(350, 60, 111, 80))
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayoutWidget_5 = QWidget(self.centralwidget)
        self.horizontalLayoutWidget_5.setObjectName(u"horizontalLayoutWidget_5")
        self.horizontalLayoutWidget_5.setGeometry(QRect(490, 60, 111, 80))
        self.horizontalLayout_5 = QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.textEdit_3 = QTextEdit(self.centralwidget)
        self.textEdit_3.setObjectName(u"textEdit_3")
        self.textEdit_3.setGeometry(QRect(190, 40, 104, 21))
        self.textEdit_4 = QTextEdit(self.centralwidget)
        self.textEdit_4.setObjectName(u"textEdit_4")
        self.textEdit_4.setGeometry(QRect(350, 40, 104, 21))
        self.textEdit_5 = QTextEdit(self.centralwidget)
        self.textEdit_5.setObjectName(u"textEdit_5")
        self.textEdit_5.setGeometry(QRect(490, 40, 104, 21))

        
        
        
        layout = QVBoxLayout()
        # 첫번째 레이아웃은 데이터 저장 위치를 입력받는 레이아웃
        self.dirInput = QLineEdit(self)
        self.dirInput.setPlaceholderText('Enter data storage location')
        layout.addWidget(self.dirInput)
        # 두번째 레이아웃은 첫번째 레이아웃 아래쪽에 위치하며 네 종류의 인풋을 입력 받으며, 수평적으로 배치


        self.Inertia_Input = QLineEdit(self)
        self.Inertia_Input.setPlaceholderText('Enter Inertia value')
        layout.addWidget(self.Inertia_Input)

        self.FP_Input = QLineEdit(self)
        self.FP_Input.setPlaceholderText('Enter First Pole value')
        layout.addWidget(self.FP_Input)

        self.SP_Input = QLineEdit(self)
        self.SP_Input.setPlaceholderText('Enter Second Pole value')
        layout.addWidget(self.SP_Input)

        self.IP_Input = QLineEdit(self)
        self.IP_Input.setPlaceholderText('Enter Integral Pole value')
        layout.addWidget(self.IP_Input)

        self.findButton = QPushButton('Find Images', self)
        self.findButton.clicked.connect(self.drawImages)
        layout.addWidget(self.findButton)

        self.leftImageLabel = QLabel(self)
        self.rightImageLabel = QLabel(self)
        layout.addWidget(self.leftImageLabel)
        layout.addWidget(self.rightImageLabel)

        self.setLayout(layout)
        '''
    def drawImages(self):
        dataDir = self.dirInput.text()
        aValue = self.aInput.text()
        bValue = self.bInput.text()

        # 이미지 찾기 로직 구현
        cSubFolder = os.path.join(dataDir, "C")
        dSubFolder = os.path.join(dataDir, "D")

        # 예시 로직: 실제 파일 이름 및 경로 패턴에 따라 조정 필요
        for subdir, dirs, files in os.walk(cSubFolder):
            for file in files:
                if file.startswith(f"{aValue}_{bValue}") and file.endswith(".png"):
                    self.leftImageLabel.setPixmap(QPixmap(os.path.join(subdir, file)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageFinderApp()
    ex.show()
    sys.exit(app.exec_())

        
'''
import functools
import os
import threading

from PyQt5 import QtCore, QtWidgets

import dash
import dash_html_components as html
from dash.dependencies import Input, Output

from PIL import Image


class MainWindow(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class Manager(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = None

    @property
    def view(self):
        return self._view

    def init_gui(self):
        self._view = MainWindow()

    @QtCore.pyqtSlot()
    def show_popup(self):
        if self.view is not None:
            self.view.show()


qt_manager = Manager()

app = dash.Dash()

app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Button("show pop up", id="button"),
        html.H2(children="", id="result"),
    ]
)


@app.callback(
    Output(component_id="result", component_property="children"),
    [Input(component_id="button", component_property="n_clicks")],
)
def popUp(n_clicks):
    if not n_clicks:
        dash.no_update

    loop = QtCore.QEventLoop()
    qt_manager.view.closed.connect(loop.quit)
    QtCore.QMetaObject.invokeMethod(
        qt_manager, "show_popup", QtCore.Qt.QueuedConnection
    )
    loop.exec()

    return "You saw a pop-up"


def main():
    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        qt_app = QtWidgets.QApplication([os.getcwd()])
    qt_app.setQuitOnLastWindowClosed(False)
    qt_manager.init_gui()
    threading.Thread(
        target=app.run_server, kwargs=dict(debug=False), daemon=True,
    ).start()

    return qt_app.exec()


if __name__ == "__main__":
    main()

'''



'''
app = dash.Dash()

app.layout = html.Div([
])
# 정보를 가져와서 저장 및 표시하는 콜백 함수 정의
@app.callback(
    [Output('image', 'src')],
    [Input('scatter-3d', 'clickData')],
    [Input(setting['id'], 'value') for setting in dropdown_settings]
)
def display_clicked_info(clickData, real_inertia, damper_ratio, integral_pole, acc_time):
    # 참조 : https://community.plotly.com/t/how-to-embed-images-into-a-dash-app/61839
    # Pilow Image 사용
    # assets directory 생성 이후 이미지 넣어두기
    clicked_info = clickData['points'][0]
    

    inertia_ratio = clickData['points'][0]['x']
    first_pole = clickData['points'][0]['y']
    second_pole = clickData['points'][0]['z']

    # 이미지 경로 생성 & Pilow Image 사용하여 Raw-Data Open 
    image_folder = f'I.R_{int(inertia_ratio):04d}_D.R_{int(damper_ratio):04d}_F.P_{int(first_pole):04d}_S.P_{int(second_pole):04d}_I.P_{int(integral_pole):04d}_A.T_{float(acc_time):.3f}_R.I_{int(real_inertia):04d}'
    try:
        image_path = os.path.join(os.path.dirname(os.path.abspath('__file__')),'data', image_folder, 'trial_0', 'raw_data_axis_2.png')
        pil_image = Image.open(image_path)
        # Image 비율을 유지한 채 resize
        pil_image.thumbnail((600, 600))
        return [pil_image]
    except:
        no_data_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'assets', 'No_Data.png')
        pil_no_image = Image.open(no_data_path)
        pil_no_image.thumbnail((600,600))
        return [pil_no_image]


def main():
    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        qt_app = QtWidgets.QApplication([os.getcwd()])

'''