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


def test_testing_tabs_are_grouped_under_parent_tab(app: MainWindow):
    assert app.tabs.count() == 6
    assert app.tabs.tabText(5) == "Testing && Experiments"
    assert app.tabs.widget(5) is app.testing_experiments_tab

    assert app.testing_experiments_tabs.count() == 4
    assert [
        app.testing_experiments_tabs.tabText(index)
        for index in range(app.testing_experiments_tabs.count())
    ] == [
        "K-Folds Testing",
        "Metamorphic Testing",
        "SADL",
        "SADL Expansion",
    ]


def test_sadl_expansion_default_portions(app: MainWindow):
    assert app.sadl_expansion_tab.initial_train_fraction.value() == 0.50
    assert app.sadl_expansion_tab.expansion_step_fraction.value() == 0.25
    assert app.sadl_expansion_tab.holdout_fraction.value() == 0.25


def test_experiment_data_root_placeholders(app: MainWindow):
    expected = "Select data root (GDXray format)"
    assert app.kfold_tab.data_root.edit.placeholderText() == expected
    assert app.sadl_expansion_tab.data_root.edit.placeholderText() == expected
