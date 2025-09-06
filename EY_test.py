import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, 
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, 
    QLabel, QFileDialog, QMessageBox, QTableWidget, 
    QProgressBar, QGroupBox, QTextEdit, QTableWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QObject
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from transformers import pipeline
from keybert import KeyBERT
import seaborn as sns
from fpdf import FPDF
import tempfile
from wordcloud import WordCloud
from collections import Counter
import re

class AIPoweredAnalyzer:
    """Advanced AI analysis engine with fallback mechanisms"""
    def __init__(self):
        self.themes = [
            "Leadership", "Technical Skills", "Team Collaboration",
            "Innovation", "Professional Conduct", "Results Achievement",
            "Communication", "Problem Solving", "Adaptability"
        ]
        self.score_map = {
            "excellent": 5, "very good": 4, "good": 3,
            "needs improvement": 2, "poor": 1, "unsatisfactory": 1
        }
        self.models = self._init_ai_models()
        
    def _init_ai_models(self):
        """Initialize AI models with proper error handling"""
        models = {}
        try:
            # Sentiment analysis
            models['sentiment'] = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Theme classification
            models['theme'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Keyword extraction
            models['keywords'] = KeyBERT("all-MiniLM-L6-v2")
            
            # Emotion detection
            models['emotion'] = pipeline(
                "text-classification", 
                model="finiteautomata/bertweet-base-emotion-analysis"
            )
            
        except Exception as e:
            print(f"AI Model Initialization Warning: {str(e)}")
            models['error'] = str(e)
        return models
    
    def analyze_employee(self, employee_data):
        """Comprehensive AI-powered analysis"""
        # Ensure we have valid data
        if employee_data.empty:
            raise ValueError("No employee data provided")
            
        # Process quantitative data
        quant_results = self._analyze_quantitative(employee_data)
        
        # Process qualitative data
        qual_results = self._analyze_qualitative(employee_data)
        
        # Generate insights
        insights = self._generate_insights(quant_results, qual_results)
        
        return {
            'employee_info': self._get_employee_info(employee_data),
            'quantitative': quant_results,
            'qualitative': qual_results,
            'insights': insights,
            'raw_data': self._prepare_raw_data_view(employee_data)
        }
    
    def _analyze_quantitative(self, data):
        """Analyze numerical feedback scores with AI-enhanced interpretation"""
        # Clean and map scores
        data['cleaned_response'] = data['Feedback Responses'].str.lower().str.strip()
        data['numeric_score'] = data['cleaned_response'].map(
            lambda x: self._match_score(x) if pd.notna(x) else np.nan
        )
        
        # Calculate basic metrics
        overall_score = data['numeric_score'].mean()
        score_dist = dict(Counter(data['cleaned_response'].dropna()))
        
        # AI-enhanced dimension analysis
        dim_scores = {}
        for dim in self.themes:
            dim_data = data[
                data['Feedback Dimensions & Questions'].str.contains(dim, case=False, na=False)
            ]
            
            if not dim_data.empty:
                dim_scores[dim] = {
                    'mean_score': dim_data['numeric_score'].mean(),
                    'feedback_count': len(dim_data),
                    'representative_feedback': self._get_representative_feedback(dim_data)
                }
        
        return {
            'overall_score': overall_score,
            'score_distribution': score_dist,
            'dimension_scores': dim_scores,
            'score_consistency': self._calculate_score_consistency(data)
        }
    
    def _match_score(self, response):
        """Enhanced score matching with regex patterns"""
        patterns = {
            r'excel.*': 5,
            r'very good': 4,
            r'good': 3,
            r'need.*improve.*': 2,
            r'poor|unsatisfactory': 1
        }
        
        for pattern, score in patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                return score
        return np.nan
    
    def _get_representative_feedback(self, dim_data):
        """Use AI to identify most representative feedback for each dimension"""
        if 'keywords' not in self.models or dim_data.empty:
            return None
            
        try:
            text_samples = dim_data['Comments'].dropna().sample(min(5, len(dim_data))).tolist()
            combined_text = " ".join(text_samples)
            keywords = self.models['keywords'].extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=3
            )
            return [kw[0] for kw in keywords]
        except:
            return None
    
    def _calculate_score_consistency(self, data):
        """Analyze how consistent scores are across different evaluators"""
        if 'Form Completed On Date' in data.columns:
            data['eval_period'] = pd.to_datetime(data['Form Completed On Date']).dt.to_period('M')
            consistency = data.groupby('eval_period')['numeric_score'].std().mean()
            return max(0, 1 - consistency/2)  # Normalize to 0-1 scale
        return None
    
    def _analyze_qualitative(self, data):
        """Advanced qualitative analysis using multiple AI techniques"""
        comments = data['Comments'].dropna()
        if comments.empty:
            return None
            
        combined_text = " ".join(comments.str.lower())
        
        results = {}
        
        # Sentiment analysis
        if 'sentiment' in self.models:
            try:
                sample_text = combined_text[:2000]  # Model limit
                sent_result = self.models['sentiment'](sample_text)
                results['sentiment'] = {
                    'label': sent_result[0]['label'],
                    'score': sent_result[0]['score']
                }
            except Exception as e:
                print(f"Sentiment analysis error: {str(e)}")
        
        # Theme analysis
        if 'theme' in self.models:
            try:
                theme_result = self.models['theme'](
                    combined_text[:2000],
                    self.themes,
                    multi_label=True
                )
                results['themes'] = {
                    label: score 
                    for label, score in zip(theme_result['labels'], theme_result['scores'])
                    if score > 0.3
                }
            except Exception as e:
                print(f"Theme analysis error: {str(e)}")
        
        # Keyword extraction
        if 'keywords' in self.models:
            try:
                keywords = self.models['keywords'].extract_keywords(
                    combined_text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    top_n=10
                )
                results['keywords'] = [kw[0] for kw in keywords]
            except Exception as e:
                print(f"Keyword extraction error: {str(e)}")
        
        # Word cloud analysis
        try:
            wordcloud = WordCloud(width=800, height=400).generate(combined_text)
            results['wordcloud'] = wordcloud
        except Exception as e:
            print(f"Word cloud error: {str(e)}")
        
        return results if results else None
    
    def _generate_insights(self, quant, qual):
        """Generate sophisticated insights combining quantitative and qualitative analysis"""
        insights = []
        
        # Score-based insights
        overall = quant.get('overall_score', 0)
        if overall >= 4:
            insights.append("Consistently high performer across all metrics")
        elif overall <= 2:
            insights.append("Significant improvement needed across multiple areas")
        
        # Dimension-based insights
        if quant.get('dimension_scores'):
            strongest = max(quant['dimension_scores'].items(), 
                          key=lambda x: x[1]['mean_score'])
            weakest = min(quant['dimension_scores'].items(), 
                         key=lambda x: x[1]['mean_score'])
            
            insights.append(f"Greatest strength in {strongest[0]} (score: {strongest[1]['mean_score']:.1f})")
            insights.append(f"Primary development area is {weakest[0]} (score: {weakest[1]['mean_score']:.1f})")
            
            if weakest[1]['mean_score'] < 2.5:
                insights.append(f"Urgent action needed to improve {weakest[0]} performance")
        
        # Sentiment-based insights
        if qual and qual.get('sentiment'):
            if qual['sentiment']['label'] == 'NEGATIVE' and qual['sentiment']['score'] > 0.7:
                insights.append("Strong negative sentiment detected in feedback comments")
            elif qual['sentiment']['label'] == 'POSITIVE' and qual['sentiment']['score'] > 0.8:
                insights.append("Overwhelmingly positive sentiment in feedback")
        
        # Theme-based insights
        if qual and qual.get('themes'):
            dominant_theme = max(qual['themes'].items(), key=lambda x: x[1])
            if dominant_theme[1] > 0.7:
                insights.append(f"Feedback strongly emphasizes {dominant_theme[0]} aspects")
        
        # Consistency insight
        if quant.get('score_consistency'):
            if quant['score_consistency'] < 0.5:
                insights.append("Inconsistent scoring patterns detected across evaluators")
        
        if not insights:
            insights.append("No strong patterns detected in the feedback data")
            
        return insights
    
    def _get_employee_info(self, data):
        """Flexible employee info extraction that works with available columns"""
        info = {
            'id': data['Feedback Requester User ID'].iloc[0]
        }
        
        # Flexible name handling
        if 'Feedback Requester First Name' in data.columns and 'Feedback Requester Last Name' in data.columns:
            info['name'] = f"{data['Feedback Requester First Name'].iloc[0]} {data['Feedback Requester Last Name'].iloc[0]}"
        elif 'Feedback Requester GUI' in data.columns:
            info['name'] = data['Feedback Requester GUI'].iloc[0]
        else:
            info['name'] = f"Employee {info['id']}"
        
        # Flexible position handling
        if 'Feedback Requester Rank' in data.columns:
            info['position'] = data['Feedback Requester Rank'].iloc[0]
        elif 'Feedback Requester Position' in data.columns:
            info['position'] = data['Feedback Requester Position'].iloc[0]
        else:
            info['position'] = "Unknown"
            
        return info
    
    def _prepare_raw_data_view(self, data):
        """Prepare a simplified view of raw data for display"""
        return data[[
            'Feedback Dimensions & Questions',
            'Feedback Responses',
            'Comments',
            'Form Completed On Date'
        ]].copy()

class HRDataManager:
    """Handles all data loading and preprocessing"""
    def __init__(self):
        self.raw_data = None
        self.employees = []
        
    def load_data(self, file_paths):
        """Load and combine data from multiple files"""
        dfs = []
        required_columns = [
            'Feedback Requester User ID',
            'Feedback Dimensions & Questions',
            'Feedback Responses',
            'Comments'
        ]
        
        for file in file_paths:
            try:
                df = pd.read_excel(file)
                
                # Normalize column names
                df.columns = df.columns.str.strip()
                
                # Convert ID column to string for consistent comparison
                if 'Feedback Requester User ID' in df.columns:
                    df['Feedback Requester User ID'] = df['Feedback Requester User ID'].astype(str).str.strip()
                
                # Check for required columns
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns in {os.path.basename(file)}: {missing}")
                
                dfs.append(df)
            except Exception as e:
                raise ValueError(f"Error loading {os.path.basename(file)}: {str(e)}")
        
        if not dfs:
            raise ValueError("No valid data files loaded")
            
        self.raw_data = pd.concat(dfs, ignore_index=True)
        self._process_data()
        return True
    
    def _process_data(self):
        """Clean and prepare the data"""
        # Ensure User ID is string type
        if 'Feedback Requester User ID' in self.raw_data.columns:
            self.raw_data['Feedback Requester User ID'] = self.raw_data['Feedback Requester User ID'].astype(str).str.strip()
            self.employees = sorted(self.raw_data['Feedback Requester User ID'].unique().tolist())
    
    def get_employee_data(self, employee_id):
        """Get all records for a specific employee"""
        if self.raw_data is None:
            raise ValueError("No data loaded")
            
        # Convert input to string and strip whitespace
        employee_id = str(employee_id).strip()
        
        employee_data = self.raw_data[
            self.raw_data['Feedback Requester User ID'].astype(str).str.strip() == employee_id
        ].copy()
        
        if employee_data.empty:
            available_ids = self.raw_data['Feedback Requester User ID'].unique()[:5]  # Show first 5 as sample
            raise ValueError(
                f"No data found for employee {employee_id}\n"
                f"Available IDs sample: {', '.join(map(str, available_ids))}"
            )
            
        return employee_data
    def _get_employee_info(self, data):
        """Extract employee information with flexible name handling"""
        info = {
            'id': data['Feedback Requester User ID'].iloc[0],
            'position': data['Feedback Requester Rank'].iloc[0] if 'Feedback Requester Rank' in data.columns else "Unknown"
        }
        
        # Flexible name handling
        if 'Feedback Requester First Name' in data.columns and 'Feedback Requester Last Name' in data.columns:
            info['name'] = f"{data['Feedback Requester First Name'].iloc[0]} {data['Feedback Requester Last Name'].iloc[0]}"
        elif 'Feedback Requester GUI' in data.columns:
            info['name'] = data['Feedback Requester GUI'].iloc[0]
        else:
            info['name'] = f"Employee {info['id']}"
            
        return info    
    def _process_data(self):
        """Clean and prepare the data"""
        # Standardize text data
        text_cols = ['Feedback Responses', 'Comments', 'Feedback Dimensions & Questions']
        for col in text_cols:
            if col in self.raw_data.columns:
                self.raw_data[col] = self.raw_data[col].astype(str).str.strip()
                
        # Extract unique employees
        if 'Feedback Requester User ID' in self.raw_data.columns:
            self.employees = sorted(self.raw_data['Feedback Requester User ID'].unique())
    
    def get_employee_data(self, employee_id):
        """Get all records for a specific employee"""
        if self.raw_data is None:
            raise ValueError("No data loaded")
            
        employee_data = self.raw_data[
            self.raw_data['Feedback Requester User ID'] == employee_id
        ].copy()
        
        if employee_data.empty:
            raise ValueError(f"No data found for employee {employee_id}")
            
        return employee_data

class AnalysisVisualizer:
    """Handles all visualization components"""
    @staticmethod
    def create_dashboard(results):
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 2)
        
        # 1. Radar Chart (Top Left)
        ax1 = fig.add_subplot(gs[0, 0], polar=True)
        AnalysisVisualizer._create_radar_chart(
            ax1,
            results['quantitative']['dimension_scores']
        )
        
        # 2. Score Distribution (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        AnalysisVisualizer._create_score_distribution(
            ax2,
            results['quantitative']['score_distribution']
        )
        
        # 3. Sentiment Analysis (Middle Left)
        ax3 = fig.add_subplot(gs[1, 0])
        AnalysisVisualizer._create_sentiment_analysis(
            ax3,
            results['qualitative']['sentiment'] if results['qualitative'] else None
        )
        
        # 4. Theme Strength (Middle Right)
        ax4 = fig.add_subplot(gs[1, 1])
        AnalysisVisualizer._create_theme_strength(
            ax4,
            results['qualitative']['themes'] if results['qualitative'] else None
        )
        
        # 5. Word Cloud (Bottom Span)
        ax5 = fig.add_subplot(gs[2, :])
        AnalysisVisualizer._create_wordcloud(
            ax5,
            results['qualitative']['wordcloud'] if results['qualitative'] else None
        )
        
        fig.suptitle(
            f"Advanced Performance Analysis\n{results['employee_info']['name']} ({results['employee_info']['position']})",
            fontsize=16,
            y=1.02
        )
        return fig
    
    @staticmethod
    def _create_radar_chart(ax, scores):
        """Create competency radar chart"""
        if not scores:
            ax.text(0.5, 0.5, "No dimension scores available", 
                   ha='center', va='center')
            return
            
        labels = list(scores.keys())
        values = [v['mean_score'] for v in scores.values()]
        
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.fill(angles, values, color='skyblue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2, marker='o')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 5.5)
        ax.grid(True)
        ax.set_title("Competency Assessment", pad=20)
    
    @staticmethod
    def _create_score_distribution(ax, distribution):
        """Create feedback score distribution chart"""
        if not distribution:
            ax.text(0.5, 0.5, "No score distribution data", 
                   ha='center', va='center')
            return
            
        labels = [k.title() for k in distribution.keys()]
        values = list(distribution.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        bars = ax.bar(labels, values, color=colors)
        
        ax.set_title("Feedback Score Distribution")
        ax.set_ylabel("Number of Responses")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
    
    @staticmethod
    def _create_sentiment_analysis(ax, sentiment):
        """Create sentiment analysis visualization"""
        if not sentiment:
            ax.text(0.5, 0.5, "No sentiment analysis data", 
                   ha='center', va='center')
            return
            
        labels = ['Negative', 'Positive']
        values = [0, 0]
        
        if sentiment['label'] == 'NEGATIVE':
            values[0] = sentiment['score'] * 100
            values[1] = (1 - sentiment['score']) * 100
        else:
            values[1] = sentiment['score'] * 100
            values[0] = (1 - sentiment['score']) * 100
            
        colors = ['#ff6b6b', '#51cf66']
        explode = (0.1, 0) if values[0] > values[1] else (0, 0.1)
        
        ax.pie(values, explode=explode, labels=labels, colors=colors,
              autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title("Feedback Sentiment Analysis")
        ax.axis('equal')
    
    @staticmethod
    def _create_theme_strength(ax, themes):
        """Create theme strength visualization"""
        if not themes:
            ax.text(0.5, 0.5, "No theme analysis data", 
                   ha='center', va='center')
            return
            
        labels = list(themes.keys())
        values = [v * 100 for v in themes.values()]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(labels)))
        y_pos = np.arange(len(labels))
        
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Strength (%)')
        ax.set_title("Dominant Themes in Feedback")
        ax.set_xlim(0, 100)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 1, i, f"{v:.1f}%", color='black', va='center')
    
    @staticmethod
    def _create_wordcloud(ax, wordcloud):
        """Create word cloud visualization"""
        if not wordcloud:
            ax.text(0.5, 0.5, "No word cloud data", 
                   ha='center', va='center')
            return
            
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title("Frequent Words in Feedback")
        ax.axis('off')

class HRAnalysisApp(QMainWindow):
    """Main application window with enhanced UI"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Powered HR Analytics Suite")
        self.resize(1400, 900)
        
        self.data_manager = HRDataManager()
        self.analyzer = AIPoweredAnalyzer()
        self.current_results = None
        
        self._init_ui()
        self._setup_connections()
    
    def _init_ui(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left panel - controls
        control_panel = QGroupBox("Analysis Controls")
        control_layout = QVBoxLayout()
        
        # Data loading section
        self.load_btn = QPushButton("Load Feedback Data")
        self.load_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        
        # Employee selection
        self.employee_combo = QComboBox()
        self.employee_combo.setPlaceholderText("Select Employee")
        
        # Analysis buttons
        self.analyze_btn = QPushButton("Run Advanced Analysis")
        self.analyze_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        
        # Export options
        self.export_btn = QPushButton("Generate Comprehensive Report")
        self.export_btn.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 8px;"
        )
        
        # Status display
        self.status_bar = QProgressBar()
        self.status_bar.setTextVisible(False)
        self.status_label = QLabel("Ready to load data")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to control panel
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(QLabel("Select Employee:"))
        control_layout.addWidget(self.employee_combo)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addWidget(self.export_btn)
        control_layout.addWidget(self.status_bar)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # Right panel - results display
        self.results_panel = QTabWidget()
        
        # Dashboard tab
        self.dashboard_tab = QWidget()
        self.dashboard_layout = QVBoxLayout()
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 8)))
        self.dashboard_layout.addWidget(self.canvas)
        self.dashboard_tab.setLayout(self.dashboard_layout)
        
        # Detailed analysis tab
        self.details_tab = QWidget()
        details_layout = QVBoxLayout()
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        details_layout.addWidget(self.analysis_text)
        self.details_tab.setLayout(details_layout)
        
        # Raw data tab
        self.raw_data_tab = QWidget()
        raw_data_layout = QVBoxLayout()
        self.raw_data_table = QTableWidget()
        self.raw_data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        raw_data_layout.addWidget(self.raw_data_table)
        self.raw_data_tab.setLayout(raw_data_layout)
        
        # Add tabs
        self.results_panel.addTab(self.dashboard_tab, "AI Dashboard")
        self.results_panel.addTab(self.details_tab, "Detailed Insights")
        self.results_panel.addTab(self.raw_data_tab, "Raw Feedback Data")
        
        # Add panels to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.results_panel)
    
    def _setup_connections(self):
        """Connect UI signals to slots"""
        self.load_btn.clicked.connect(self.load_data)
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.export_btn.clicked.connect(self.export_report)
    
    def load_data(self):
        """Load evaluation data from files"""
        # Get files from file dialog
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Feedback Data Files",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if not selected_files:
            return  # User cancelled the dialog
            
        try:
            self.status_label.setText("Loading data...")
            self.status_bar.setRange(0, 0)
            QApplication.processEvents()
            
            # Use selected_files instead of undefined 'files' variable
            success = self.data_manager.load_data(selected_files)
            if success:
                self.employee_combo.clear()
                self.employee_combo.addItems([str(e) for e in self.data_manager.employees])
                self.status_label.setText(
                    f"Loaded {len(self.data_manager.employees)} employees"
                )
                QMessageBox.information(
                    self,
                    "Data Loaded",
                    f"Successfully loaded feedback data for {len(self.data_manager.employees)} employees"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Loading Error",
                f"Failed to load data:\n{str(e)}"
            )
        finally:
            self.status_bar.setRange(0, 100)
        
    def run_analysis(self):
        """Run analysis for selected employee"""
        employee_id = self.employee_combo.currentText()
        if not employee_id:
            QMessageBox.warning(self, "Warning", "Please select an employee")
            return
            
        try:
            self._set_ui_enabled(False)
            self.status_label.setText(f"Analyzing employee {employee_id}...")
            self.status_bar.setRange(0, 0)
            QApplication.processEvents()
            
            # Get employee data
            employee_data = self.data_manager.get_employee_data(employee_id)
            
            # Run analysis
            results = self.analyzer.analyze_employee(employee_data)
            self.current_results = results
            
            # Update UI with results
            self._display_results(results)
            
            self.status_label.setText("Analysis complete")
            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Completed analysis for employee {employee_id}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Analysis Error",
                f"Failed to analyze employee:\n{str(e)}"
            )
            print(f"Analysis error: {str(e)}")
        finally:
            self._set_ui_enabled(True)
            self.status_bar.setRange(0, 100)
    
    def _display_results(self, results):
        """Display analysis results across all tabs"""
        # Update dashboard
        self.canvas.figure = AnalysisVisualizer.create_dashboard(results)
        self.canvas.draw()
        
        # Update detailed insights
        self.analysis_text.setHtml(self._format_insights_text(results))
        
        # Update raw data table
        self._display_raw_data(results['raw_data'])
    
    def _format_insights_text(self, results):
        """Format analysis results as HTML"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial; margin: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                h2 {{ color: #2980b9; }}
                h3 {{ color: #16a085; }}
                .highlight {{ background-color: #fffde7; padding: 2px 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th {{ background-color: #3498db; color: white; text-align: left; padding: 8px; }}
                td {{ border: 1px solid #ddd; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .neutral {{ color: #f39c12; }}
            </style>
        </head>
        <body>
            <h1>Advanced Employee Performance Analysis</h1>
            <h2>{results['employee_info']['name']} (ID: {results['employee_info']['id']})</h2>
            <h3>{results['employee_info']['position']}</h3>
            <h3>Executive Summary</h3>
            <p>Overall Performance Score: <span class="highlight">{results['quantitative']['overall_score']:.1f}/5</span></p>
            <h3>Key Insights</h3>
            <ul>
                {''.join(f"<li>{insight}</li>" for insight in results['insights'])}
            </ul>
            <h3>Competency Breakdown</h3>
            <table>
                <tr>
                    <th>Competency Area</th>
                    <th>Average Score</th>
                    <th>Feedback Count</th>
                    <th>Key Themes</th>
                </tr>
                {''.join(
                    f'''
                    <tr>
                        <td>{dim}</td>
                        <td>{scores['mean_score']:.1f}</td>
                        <td>{scores['feedback_count']}</td>
                        <td>{', '.join(scores['representative_feedback']) if scores['representative_feedback'] else "N/A"}</td>
                    </tr>
                    ''' for dim, scores in results['quantitative']['dimension_scores'].items()
                )}
            </table>
            <h3>Qualitative Analysis</h3>
            {''.join(
                f'''
                <h4>{section.replace('_', ' ').title()}</h4>
                <p>{str(results['qualitative'][section])}</p>
                ''' for section in results['qualitative'] if results['qualitative'][section]
            )}
        </body>
        </html>
        """
        return html

# The line below was also incomplete in your code:
# code, fname = *get*code_from_file(run_name, path_name)
# It should be something like:
# code, fname = get_code_from_file(run_name, pa
    
    def _display_raw_data(self, raw_data):
        """Display raw data in table format"""
        self.raw_data_table.clear()
        
        # Set up table
        self.raw_data_table.setRowCount(len(raw_data))
        self.raw_data_table.setColumnCount(len(raw_data.columns))
        self.raw_data_table.setHorizontalHeaderLabels(raw_data.columns)
        
        # Populate table
        for row_idx, row in raw_data.iterrows():
            for col_idx, col in enumerate(raw_data.columns):
                item = QTableWidgetItem(str(row[col]))
                self.raw_data_table.setItem(row_idx, col_idx, item)
        
        # Resize columns
        self.raw_data_table.resizeColumnsToContents()
    
    def export_report(self):
        """Export comprehensive analysis report"""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No analysis results to export")
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Analysis Report",
                "",
                "PDF Files (*.pdf)"
            )
            
            if file_path:
                self.status_label.setText("Generating report...")
                self.status_bar.setRange(0, 0)
                QApplication.processEvents()
                
                # Create PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Add title and metadata
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="Employee Performance Analysis Report", ln=1, align='C')
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Employee: {self.current_results['employee_info']['name']}", ln=1)
                pdf.cell(200, 10, txt=f"Position: {self.current_results['employee_info']['position']}", ln=1)
                pdf.cell(200, 10, txt=f"Employee ID: {self.current_results['employee_info']['id']}", ln=1)
                pdf.cell(200, 10, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
                pdf.ln(10)
                
                # Add summary section
                # Add summary section
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Executive Summary", ln=1)
                pdf.set_font("Arial", size=12)

                # Prepare insights text first
                insights_text = "\n".join([f"- {insight}" for insight in self.current_results['insights']])

                # Single multi_cell call with properly formatted text
                pdf.multi_cell(0, 10, txt=f"""Overall Performance Score: {self.current_results['quantitative']['overall_score']:.1f}/5

                Key Insights:
                {insights_text}""")
                                
                # Save visualizations as temporary images
                img_paths = []
                try:
                    # Dashboard image
                    dashboard_path = os.path.join(tempfile.gettempdir(), "dashboard.png")
                    self.canvas.figure.savefig(dashboard_path, bbox_inches='tight', dpi=150)
                    img_paths.append(dashboard_path)
                    
                    # Add visualizations to PDF
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Performance Dashboard", ln=1)
                    pdf.image(dashboard_path, x=10, y=30, w=180)
                    
                    # Add detailed findings
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Detailed Findings", ln=1)
                    pdf.set_font("Arial", size=12)
                    
                    # Competency breakdown
                    pdf.cell(200, 10, txt="Competency Breakdown:", ln=1)
                    for dim, scores in self.current_results['quantitative']['dimension_scores'].items():
                        pdf.multi_cell(0, 10, txt=f"""
{dim}:
- Average Score: {scores['mean_score']:.1f}/5
- Feedback Count: {scores['feedback_count']}
- Key Themes: {", ".join(scores['representative_feedback']) if scores['representative_feedback'] else "N/A"}
                        """)
                    
                    # Qualitative findings
                    pdf.ln(5)
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Qualitative Analysis", ln=1)
                    pdf.set_font("Arial", size=12)
                    
                    if self.current_results['qualitative']:
                        for section, content in self.current_results['qualitative'].items():
                            if content:
                                pdf.multi_cell(0, 10, txt=f"""
{section.replace('_', ' ').title()}:
{str(content)}
                                """)
                    
                finally:
                    # Clean up temporary files
                    for path in img_paths:
                        try:
                            os.remove(path)
                        except:
                            pass
                
                # Save PDF
                pdf.output(file_path)
                
                self.status_label.setText("Report generated successfully")
                QMessageBox.information(
                    self,
                    "Report Generated",
                    f"Successfully saved report to:\n{file_path}"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Report Error",
                f"Failed to generate report:\n{str(e)}"
            )
            print(f"Report generation error: {str(e)}")
        finally:
            self.status_bar.setRange(0, 100)
    
    def _set_ui_enabled(self, enabled):
        """Enable/disable UI elements during processing"""
        self.load_btn.setEnabled(enabled)
        self.employee_combo.setEnabled(enabled)
        self.analyze_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = HRAnalysisApp()
    window.show()
    sys.exit(app.exec())
