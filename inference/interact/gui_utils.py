from functools import partial
from typing import Optional, Union
import time
import traceback, sys
from PyQt5 import QtCore
from PyQt5.QtGui import QPalette, QColor

from PyQt5.QtCore import Qt, QRunnable, pyqtSlot, pyqtSignal, QObject, QPoint, QRect, QSize
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QProgressBar, QDialog, QWidget,
                             QProgressDialog, QScrollArea, QLayout, QLayoutItem, QStyle, QSizePolicy, QSpacerItem,
                                QFrame, QPushButton, QSlider, QMessageBox, QGridLayout)

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


def create_parameter_box(min_val, max_val, text, step=1, callback=None):
    layout = QHBoxLayout()

    dial = QSpinBox()
    dial.setMaximumHeight(28)
    dial.setMaximumWidth(150)
    dial.setMinimum(min_val)
    dial.setMaximum(max_val)
    dial.setAlignment(Qt.AlignRight)
    dial.setSingleStep(step)
    dial.valueChanged.connect(callback)

    label = QLabel(text)
    label.setAlignment(Qt.AlignRight)

    layout.addWidget(label)
    layout.addWidget(dial)

    return dial, layout


def create_gauge(text):
    layout = QHBoxLayout()

    gauge = QProgressBar()
    gauge.setMaximumHeight(28)
    gauge.setMaximumWidth(200)
    gauge.setAlignment(Qt.AlignCenter)

    label = QLabel(text)
    label.setAlignment(Qt.AlignRight)

    layout.addWidget(label)
    layout.addWidget(gauge)

    return gauge, layout


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget=None, margin: int=-1, hSpacing: int=-1, vSpacing: int=-1):
        super().__init__(parent)

        self.itemList = list()
        self.m_hSpace = hSpacing
        self.m_vSpace = vSpacing

        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        # copied for consistency, not sure this is needed or ever called
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item: QLayoutItem):
        self.itemList.append(item)

    def horizontalSpacing(self) -> int:
        if self.m_hSpace >= 0:
            return self.m_hSpace
        else:
            return self.smartSpacing(QStyle.PM_LayoutHorizontalSpacing)

    def verticalSpacing(self) -> int:
        if self.m_vSpace >= 0:
            return self.m_vSpace
        else:
            return self.smartSpacing(QStyle.PM_LayoutVerticalSpacing)

    def count(self) -> int:
        return len(self.itemList)

    def itemAt(self, index: int) -> Union[QLayoutItem, None]:
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        else:
            return None

    def takeAt(self, index: int) -> Union[QLayoutItem, None]:
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        else:
            return None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def smartSpacing(self, pm: QStyle.PixelMetric) -> int:
        parent = self.parent()
        if not parent:
            return -1
        elif parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        else:
            return parent.spacing()

    def doLayout(self, rect: QRect, testOnly: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        effectiveRect = rect.adjusted(+left, +top, -right, -bottom)
        x = effectiveRect.x()
        y = effectiveRect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.horizontalSpacing()
            if spaceX == -1:
                spaceX = wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Horizontal)
            spaceY = self.verticalSpacing()
            if spaceY == -1:
                spaceY = wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical)

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > effectiveRect.right() and lineHeight > 0:
                x = effectiveRect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
    
            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y() + bottom
    

class JFlowLayout(FlowLayout):
    # flow layout, similar to an HTML `<DIV>`
    # this is our "wrapper" to the `FlowLayout` sample Qt code we have implemented
    # we use it in place of where we used to use a `QHBoxLayout`
    # in order to make few outside-world changes, and revert to `QHBoxLayout`if we ever want to,
    # there are a couple of methods here which are available on a `QBoxLayout` but not on a `QLayout`
    # for which we provide a "lite-equivalent" which will suffice for our purposes

    def addLayout(self, layout: QLayout, stretch: int=0):
        # "equivalent" of `QBoxLayout.addLayout()`
        # we want to add sub-layouts (e.g. a `QVBoxLayout` holding a label above a widget)
        # there is some dispute as to how to do this/whether it is supported by `FlowLayout`
        # see my https://forum.qt.io/topic/104653/how-to-do-a-no-break-qhboxlayout
        # there is a suggestion that we should not add a sub-layout but rather enclose it in a `QWidget`
        # but since it seems to be working as I've done it below I'm elaving it at that for now...

        # suprisingly to me, we do not need to add the layout via `addChildLayout()`, that seems to make no difference
        # self.addChildLayout(layout)
        # all that seems to be reuqired is to add it onto the list via `addItem()`
        self.addItem(layout)

    def addStretch(self, stretch: int=0):
        # "equivalent" of `QBoxLayout.addStretch()`
        # we can't do stretches, we just arbitrarily put in a "spacer" to give a bit of a gap
        w = stretch * 20
        spacerItem = QSpacerItem(w, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.addItem(spacerItem)


class NamedSlider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, name: str, min_: int, max_: int, step_size: int, default: int, multiplier=1, min_text=None, max_text=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.multiplier = multiplier
        self.min_text = min_text
        self.max_text = max_text

        layout = QHBoxLayout(self)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_)
        self.slider.setMaximum(max_)
        self.slider.setValue(default)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(step_size)

        name_label = QLabel(name + " |")
        self.value_label = QLabel()

        layout.addWidget(name_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.slider)

        self.update_name()

        self.slider.valueChanged.connect(self.on_slide)

    def value(self):
        return self.slider.value() * self.multiplier
    
    def on_slide(self):
        self.update_name()
        self.valueChanged.emit(self.slider.value() * self.multiplier)

    def update_name(self):
        value = self.value()
        value_str = None
        if self.multiplier != 1:
            if isinstance(self.multiplier, float):
                min_str = f'{self.slider.minimum() * self.multiplier:.2f}'
                value_str = f'{value:.2f}'
                max_str = f'{self.slider.maximum() * self.multiplier:.2f}'
        
        if value_str is None:
            min_str = f'{self.slider.minimum() * self.multiplier:d}'
            value_str = f'{value:d}'
            max_str = f'{self.slider.maximum() * self.multiplier:d}'

        if self.min_text is not None:
            min_str += f' ({self.min_text})'  
        if self.max_text is not None:
            max_str += f' ({self.max_text})' 

        final_str = f'{min_str} <= {value_str} <= {max_str}' 

        self.value_label.setText(final_str)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mouseReleaseEvent(self, event):
        super(ClickableLabel, self).mousePressEvent(event)
        if event.button() == Qt.LeftButton and event.pos() in self.rect():
            self.clicked.emit()


class ImageWithCaption(QWidget):
    def __init__(self, img: QLabel, caption: str, on_close: callable = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.layout = QVBoxLayout(self)
        self.text_label = QLabel(caption)
        self.close_btn = QPushButton("x")
        self.close_btn.setMaximumSize(35, 35)
        self.close_btn.setMinimumSize(35, 35)
        self.close_btn.setStyleSheet('QPushButton {background-color: #DC4C64; font-weight: bold; }')
        if on_close is not None:
            self.close_btn.clicked.connect(on_close)

        self.top_tab_layout = QHBoxLayout()
        self.top_tab_layout.addWidget(self.text_label)
        self.top_tab_layout.addWidget(self.close_btn)
        self.top_tab_layout.setAlignment(self.text_label, Qt.AlignmentFlag.AlignCenter)
        self.top_tab_layout.setAlignment(self.close_btn, Qt.AlignmentFlag.AlignRight)
        
        self.layout.addLayout(self.top_tab_layout)

        self.layout.addWidget(img)

        self.layout.setAlignment(self.text_label, Qt.AlignmentFlag.AlignHCenter)

class ImageLinkCollection(QWidget):
    def __init__(self, on_click: callable, load_image: callable, delete_image: callable = None, name: str = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.on_click = on_click
        self.load_image = load_image
        self.delete_image = delete_image
        self.name = name
        # scrollable_area = QScrollArea(self)
        # frame = QFrame(scrollable_area)

        self.flow_layout = JFlowLayout(self)

        self.img_widgets_lookup = dict()


    def add_image(self, img_idx):
        image = self.load_image(img_idx)

        img_widget = ClickableLabel()
        img_widget.setPixmap(image)

        img_widget.clicked.connect(partial(self.on_click, img_idx))

        wrapper = ImageWithCaption(img_widget, f"Frame {img_idx:>6d}", on_close=partial(self.on_close_click, img_idx))
        # layout.addWidget(img_widget)

        self.img_widgets_lookup[img_idx] = wrapper
        self.flow_layout.addWidget(wrapper)

    def remove_image(self, img_idx):
        img_widget = self.img_widgets_lookup.pop(img_idx)
        self.flow_layout.removeWidget(img_widget)

    def on_close_click(self, img_idx):
        qm = QMessageBox(QMessageBox.Icon.Warning, "Confirm deletion", "")
        question = f"Delete Frame {img_idx}"
        if self.name is not None:
            question += f' from {self.name}'
        
        question += '?'
        ret = qm.question(self, 'Confirm deletion', question, qm.Yes | qm.No)

        if ret == qm.Yes:
            self.remove_image(img_idx)
            if self.delete_image is not None:
                self.delete_image(img_idx)

class ColorPicker(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, num_colors, color_palette: bytes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_colors = num_colors

        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.inner_layout = QGridLayout()  # 2 x N/2
        # self.inner_layout_wrapper = QHBoxLayout()
        self.inner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.palette = color_palette
        self.previously_selected = None

        for i in range(self.num_colors):
            index = i + 1

            color_widget = ClickableLabel(str(index))

            color = self.palette[index * 3: index*3 + 3] 

            color_widget.setStyleSheet(f"QLabel {{font-family: Monospace; color:white; font-weight: 900; background-color: rgb{tuple(color)}}} QLabel.selected {{border: 4px solid}}")
            color_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

            color_widget.setFixedSize(40, 40)
            self.inner_layout.addWidget(color_widget, int(i / 2), i % 2)

            color_widget.clicked.connect(partial(self._on_color_clicked, index))

        color_picker_name = QLabel("Object selector")
        color_picker_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color_picker_name.setStyleSheet("QLabel {font-family: Monospace; font-weight: 900}")

        num_objects_label = QLabel(f"({self.num_colors} objects)")
        num_objects_label.setStyleSheet("QLabel {font-family: Monospace; font-weight: 900}")
        num_objects_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        
        color_picker_instruction = QLabel("Click or use\nnumerical keys")
        color_picker_instruction.setStyleSheet("QLabel {font-family: Monospace; font-style: italic}")
        color_picker_instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text_wrapper_widget = QWidget()
        text_wrapper_widget.setStyleSheet("QWidget {background-color: rgb(225, 225, 225);}")
        text_layout = QVBoxLayout(text_wrapper_widget)
        text_layout.addWidget(color_picker_name)
        text_layout.addWidget(num_objects_label)
        text_layout.addWidget(color_picker_instruction)

        self.outer_layout.addWidget(text_wrapper_widget)
        self.outer_layout.addLayout(self.inner_layout)

        self.select(1)  # First object selected by default

    def _on_color_clicked(self, index: int):
        self.clicked.emit(index)
        pass

    def select(self, index: int):   # 1-based, not 0-based
        widget = self.inner_layout.itemAt(index - 1).widget()
        widget.setProperty("class", "selected")
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

        # print(widget.text())
        # print(widget.styleSheet())

        if self.previously_selected is not None:
            self.previously_selected.setProperty("class", "")
            self.previously_selected.style().unpolish(self.previously_selected)
            self.previously_selected.style().polish(self.previously_selected)
            self.previously_selected.update()

        self.previously_selected = self.inner_layout.itemAt(index - 1).widget()
