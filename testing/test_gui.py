import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PySide6.QtCore import Qt
from main import MainWindow


@pytest.fixture
def app(qtbot):
    test_app = MainWindow()
    qtbot.addWidget(test_app)
    return test_app


def test_first_tab(app: MainWindow, qtbot):
    """Tests if the first tab is able to be selected"""
    assert app.tabs.currentIndex() == 0

    tab_bar = app.tabs.tabBar()
    qtbot.mouseClick(tab_bar, Qt.LeftButton, pos=tab_bar.tabRect(1).center())

    assert app.tabs.currentIndex() == 1


def test_second_tab(app: MainWindow, qtbot):
    """Tests if the second tab is able to be selected"""
    assert app.tabs.currentIndex() == 0

    tab_bar = app.tabs.tabBar()
    qtbot.mouseClick(tab_bar, Qt.LeftButton, pos=tab_bar.tabRect(2).center())

    assert app.tabs.currentIndex() == 2
