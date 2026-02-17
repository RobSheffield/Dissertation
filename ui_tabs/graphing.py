from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from typing import Dict, List, Tuple

# filepath: c:/Users/Rob/Documents/Dissertation/take2/X-Ray_Image_Analysis/ui_tabs/graphing.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class MetamorphicTestGraphing:
    """Helper class for generating graphs from metamorphic testing results."""
    
    
    def plot_relation_accuracy(relation_name: str, accuracies: List[float], 
                              image_names: List[str] = None) -> Figure:
        """
        Plot accuracy results for a single metamorphic relation.
        
        Args:
            relation_name: Name of the metamorphic relation
            accuracies: List of accuracy percentages for each test image
            image_names: Optional list of image names for x-axis labels
        
        Returns:
            matplotlib Figure object
        """
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        x_pos = np.arange(len(accuracies))
        colors = ['green' if acc >= 80 else 'orange' if acc >= 50 else 'red' 
                 for acc in accuracies]
        
        ax.bar(x_pos, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Metamorphic Relation: {relation_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        if image_names and len(image_names) == len(accuracies):
            ax.set_xticks(x_pos)
            ax.set_xticklabels(image_names, rotation=45, ha='right')
        else:
            ax.set_xlabel('Test Image', fontsize=12)
            ax.set_xticks(x_pos)
        
        fig.tight_layout()
        return fig

    def plot_relation_comparison(relations_data: Dict[str, List[float]]):
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        names = list(relations_data.keys())
        values = [v[0] for v in relations_data.values()]
        
        colors = ['#2ecc71' if v > 80 else '#e67e22' if v > 50 else '#e74c3c' for v in values]
        bars = ax.bar(names, values, color=colors)
        
        # Rotate labels to prevent overlap
        ax.tick_params(axis='x', rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
            tick.set_fontsize(9)

        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Metamorphic Relation Performance")
        
        fig.tight_layout()
        return fig

    
    def plot_box_plot_comparison(results: Dict[str, List[float]]) -> Figure:
        """
        Plot box plot showing distribution of accuracies across relations.
        
        Args:
            results: Dictionary with relation names as keys and list of accuracies as values
        
        Returns:
            matplotlib Figure object
        """
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        relation_names = list(results.keys())
        accuracies_list = [results[name] for name in relation_names]
        
        bp = ax.boxplot(accuracies_list, labels=relation_names, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Metamorphic Relations', fontsize=12)
        ax.set_title('Distribution of Accuracy Across Relations', 
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels(relation_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
        return fig

    
    def plot_heatmap(results: Dict[str, List[float]], image_names: List[str] = None) -> Figure:
        """
        Plot heatmap showing accuracy for each relation across all test images.
        
        Args:
            results: Dictionary with relation names as keys and list of accuracies as values
            image_names: Optional list of image names for column labels
        
        Returns:
            matplotlib Figure object
        """
        fig = Figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        
        relation_names = list(results.keys())
        data = np.array([results[name] for name in relation_names])
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        ax.set_yticks(np.arange(len(relation_names)))
        ax.set_yticklabels(relation_names)
        
        num_images = data.shape[1]
        if image_names and len(image_names) == num_images:
            ax.set_xticks(np.arange(num_images))
            ax.set_xticklabels(image_names, rotation=45, ha='right')
        else:
            ax.set_xticks(np.arange(num_images))
            ax.set_xticklabels([f'Img {i+1}' for i in range(num_images)])
        
        ax.set_title('Accuracy Heatmap - Relations vs Test Images', 
                    fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(relation_names)):
            for j in range(num_images):
                text = ax.text(j, i, f'{data[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
        
        fig.tight_layout()
        return fig

    
    def plot_summary_statistics(results: Dict[str, List[float]]) -> Figure:
        """
        Plot summary statistics for all relations.
        
        Args:
            results: Dictionary with relation names as keys and list of accuracies as values
        
        Returns:
            matplotlib Figure object
        """
        fig = Figure(figsize=(12, 8))
        
        relation_names = list(results.keys())
        return fig

    
    def plot_perturbation_impact(perturbation_data: Dict[str, List[Tuple[float, float]]]) -> Figure:
        """
        Plot how accuracy changes with different levels of perturbation.
        X-axis is normalized to 'Severity Level' (1, 2, 3...) to allow comparison.
        
        Args:
            perturbation_data: Dict where key is relation name, 
                             value is list of (level, accuracy) tuples.
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        for relation, data in perturbation_data.items():
            if not data: continue
            # Sort data by perturbation level
            sorted_data = sorted(data, key=lambda x: x[0])
            levels, accuracies = zip(*sorted_data)
            
            # Normalize X-axis to steps (1, 2, 3, 4) since raw values are on different scales
            steps = np.arange(1, len(accuracies) + 1)
            
            ax.plot(steps, accuracies, marker='o', label=relation, linewidth=2, markersize=8)
            
        ax.set_xlabel('Severity Level (Incremental Steps)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Impact of Perturbation Severity on Model Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticks(np.arange(1, 6)) # Typically 4 steps, allow room for title
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower left', fontsize=10)
        
        fig.tight_layout()
        return fig
        stats_data = {
            'Mean': [np.mean(accuracies) for accuracies in results.values()],
            'Median': [np.median(accuracies) for accuracies in results.values()],
            'Min': [np.min(accuracies) for accuracies in results.values()],
            'Max': [np.max(accuracies) for accuracies in results.values()],
        }
        
        ax = fig.add_subplot(111)
        x = np.arange(len(relation_names))
        width = 0.2
        
        for idx, (stat_name, values) in enumerate(stats_data.items()):
            offset = (idx - 1.5) * width
            ax.bar(x + offset, values, width, label=stat_name, alpha=0.8)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Metamorphic Relations', fontsize=12)
        ax.set_title('Summary Statistics Across All Relations', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(relation_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        return fig


class GraphingCanvas(QWidget):
    """Widget to display matplotlib figures in Qt."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = None
    
    def display_figure(self, fig: Figure):
        """Display a matplotlib figure on the canvas."""
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        
        self.canvas = FigureCanvas(fig)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()