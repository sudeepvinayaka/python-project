import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, MetaData


class DataProcessorBase:
    def __init__(self, train_table, ideal_table, test_table, database_url):
        self.train_data = self.load_data_from_sql(train_table, database_url)
        self.ideal_data = self.load_data_from_sql(ideal_table, database_url)
        self.test_data = self.load_data_from_sql(test_table, database_url)

    def load_data_from_sql(self, table_name, database_url):
        engine = create_engine(database_url)
        with engine.connect() as conn:
            data = pd.read_sql_table(table_name, conn)
        return data

    def preprocess_data(self, data):
        data = data.dropna()
        normalized_data = (data - data.mean()) / data.std()
        return normalized_data

    def calculate_squared_errors(self, column1, column2):
        if len(column1) != len(column2):
            raise ValueError("Columns must have the same length.")
        squared_errors = (np.array(column1) - np.array(column2))**2
        return np.sum(squared_errors)

    def find_best_fit_column(self, normalized_train, normalized_ideal):
        best_fit_indices = []
        for train_col_index in range(1, len(normalized_train.columns)):
            train_column = normalized_train.iloc[:, train_col_index]
            sum_squared_errors_list = []

            for ideal_col_index in range(1, len(normalized_ideal.columns)):
                ideal_column = normalized_ideal.iloc[:, ideal_col_index]
                sum_squared_errors = self.calculate_squared_errors(train_column, ideal_column)
                sum_squared_errors_list.append(round(sum_squared_errors, 5))

            best_fit_index = np.argmin(sum_squared_errors_list)
            best_fit_indices.append(best_fit_index)
        return np.array(best_fit_indices) + 1  # 1-based indexing

    def plot_data(self, x_axis_range, normalized_ideal, normalized_train, best_fit_indices):
        for i in range(4):  # Assuming 4 y columns
            plt.title(f'Y{best_fit_indices[i]}')
            plt.scatter(x_axis_range, normalized_ideal.iloc[:, best_fit_indices[i]], label='Ideal')
            plt.scatter(x_axis_range, normalized_train.iloc[:, i + 1], label='Train')
            plt.legend()
            plt.show()

    def calculate_min_distance(self, array_a, value_b):
        distances = np.abs(array_a - value_b)
        return np.min(distances)


class DerivedDataProcessor(DataProcessorBase):
    def __init__(self, train_table, ideal_table, test_table, database_url):
        super().__init__(train_table, ideal_table, test_table, database_url)
        self.normalized_train_data = None
        self.normalized_ideal_data = None
        self.normalized_test_data = None
        self.best_fit_indices = None

    def classify_test_data(self, ideal_data_columns):
        distances = []
        indices = []

        for _, test_val in self.normalized_test_data.iterrows():
            min_distances = []
            for i in range(4):
                min_dist = self.calculate_min_distance(ideal_data_columns.iloc[:, i], test_val[1])
                min_distances.append(min_dist)

            distances.append(round(min(min_distances), 3))
            indices.append(np.argmin(min_distances))

        classified_labels = ['Y' + str(self.best_fit_indices[i]) for i in indices]
        return classified_labels, distances

    def main(self):
        self.normalized_train_data = self.preprocess_data(self.train_data)
        self.normalized_ideal_data = self.preprocess_data(self.ideal_data)
        self.normalized_test_data = self.preprocess_data(self.test_data)

        self.best_fit_indices = self.find_best_fit_column(self.normalized_train_data, self.normalized_ideal_data)

        x_axis_range = range(len(self.normalized_train_data))
        self.plot_data(x_axis_range, self.normalized_ideal_data, self.normalized_train_data, self.best_fit_indices)

        classified_labels, deviations = self.classify_test_data(self.normalized_ideal_data.iloc[:, self.best_fit_indices])

        final_test_data = self.test_data.copy()
        final_test_data["No. of ideal func"] = classified_labels
        final_test_data["Delta Y (test func)"] = deviations
        return final_test_data


if __name__ == "__main__":
    database_url = "sqlite:///database.db"
    processor = DerivedDataProcessor("train_table", "ideal_table", "test_table", database_url)
    final_result = processor.main()
    print(final_result)

    # Store result in DB
    engine = create_engine(database_url)
    final_result.to_sql("result", engine, index=False, if_exists="replace")
    print("✔️ Final result loaded into 'result' table in database.db")



