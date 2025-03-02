import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel
from PySide6.QtCore import Qt
import pandas as pd
from EY_test import get_employee_data, analyze_qa_answers  # Import the logic from the first script

class EmployeeSearchApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Employee Search")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Employee ID input
        self.id_input = QLineEdit(self)
        self.id_input.setPlaceholderText("Enter Employee ID")
        layout.addWidget(self.id_input)

        # Search button
        self.search_button = QPushButton("Search by ID", self)
        self.search_button.clicked.connect(self.search_employee)
        layout.addWidget(self.search_button)

        # Result area
        self.result_label = QLabel("", self)
        layout.addWidget(self.result_label)

        # Analysis buttons
        self.analysis_button = QPushButton("Analyze QA", self)
        self.analysis_button.clicked.connect(self.analyze_qa)
        layout.addWidget(self.analysis_button)

        self.comments_button = QPushButton("Analyze Comments", self)
        self.comments_button.clicked.connect(self.analyze_comments)
        layout.addWidget(self.comments_button)

        self.setLayout(layout)

    def search_employee(self):
        """Fetch employee data."""
        employee_id = self.id_input.text()
        text_df, qa_df, error = get_employee_data(employee_id)
        if error:
            self.result_label.setText(error)
        else:
            # Handle missing values in qa_df
            qa_df = qa_df.fillna("")  # Replace NaN with empty strings
            
            self.employee_data = (text_df, qa_df)
            self.result_label.setText(f"Employee {employee_id} data found.\nReady for analysis.")

    def analyze_qa(self):
        """Analyze QA answers."""
        if hasattr(self, 'employee_data'):
            text_df, qa_df = self.employee_data
            
            try:
                analysis_result = analyze_qa_answers(qa_df)
                self.result_label.setText(analysis_result)
            except Exception as e:
                self.result_label.setText(f"Error in analysis: {e}")
        else:
            self.result_label.setText("Please search for an employee first.")

    def analyze_comments(self):
        """Placeholder for comment analysis."""
        if hasattr(self, 'employee_data'):
            text_df, _ = self.employee_data
            self.result_label.setText("Comments Analysis: Placeholder for future.")
        else:
            self.result_label.setText("Please search for an employee first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmployeeSearchApp()
    window.show()
    sys.exit(app.exec())
