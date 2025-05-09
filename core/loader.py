import re
import pandas as pd

class CorpusLoader:
    """
    Loads, cleans, and formats text-label datasets from CSV files.
    Standardizes columns to ['sentence', 'label'] for embedding and projection.
    """
    def __init__(self, path=None, text_col="sentence", label_col="label"):
        self.path = path
        self.text_col = text_col
        self.label_col = label_col
        self.hugginface = None
        self.df = None
        self.embedding = None
        self.df_cache = None

    def load_csv(self, start=0, n_rows=None):
        """
        Loads a dataset from the specified path, starting at `start` and 
        loading up to `n_rows` rows.

        :param start: int
            Row index to start loading from (default is 0).
        :param n_rows: int, optional
            Number of rows to load starting from `start`. If None, load to end of file.
        :return: DataFrame
            Trimmed DataFrame with renamed 'sentence' and 'label' columns.
        """
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.path}")
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except Exception as e:
            raise RuntimeError(f"Error loading CSV: {str(e)}")

        if start >= len(df):
            raise IndexError(f"Start index {start} is beyond dataset length {len(df)}.")

        end = start + n_rows if n_rows is not None else len(df)
        df = df.iloc[start:end].reset_index(drop=True)

        # Check columns
        if self.text_col not in df.columns or self.label_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{self.text_col}' and '{self.label_col}'.")

        # Rename columns
        df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'}, inplace=True)
        self.text_col = 'sentence'
        self.label_col = 'label'
        self.df = df
        return df

    def load_from_huggingface(self, dataset_name, split="train"):
        """
        Loads a dataset from Hugging Face's datasets library.

        :param dataset_name: str
            The name of the dataset to load.
        :param split: str
            The split of the dataset to load (default is "train").
        :return: DataFrame
            The loaded DataFrame.
        """
        from datasets import load_dataset

        try:
            self.hugginface = load_dataset(dataset_name, split=split)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from Hugging Face: {str(e)}")

        if self.text_col not in self.hugginface.column_names or self.label_col not in self.hugginface.column_names:
            raise ValueError(f"Dataset must contain columns '{self.text_col}' and '{self.label_col}'.")

        # Convert to DataFrame
        df = pd.DataFrame(self.hugginface)
        
        # Rename columns
        df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'}, inplace=True)
        self.text_col = 'sentence'
        self.label_col = 'label'
        self.df = df
        return df


    def load_from_dataframe(self, df):
        """
        Loads a DataFrame directly into the CorpusLoader.

        :param df: DataFrame
            The DataFrame to load.
        :return: DataFrame
            The loaded DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        
        self.df = df.copy()
        if self.text_col not in self.df.columns or self.label_col not in self.df.columns:
            raise ValueError(f"DataFrame must contain columns '{self.text_col}' and '{self.label_col}'.")
        
        # Rename columns
        self.df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'}, inplace=True)
        self.text_col = 'sentence'
        self.label_col = 'label'
        return self.df

    def _clean_string(self, raw_string):
        """Cleans the string by removing HTML, slashes, special chars, and lowercasing."""
        if not isinstance(raw_string, str):
            raw_string = str(raw_string) if pd.notnull(raw_string) else ""
        clean_text = re.sub(r'<br\s*/?>', '', raw_string)  # Handles <br> tags
        clean_text = re.sub(r'/', ' ', clean_text)
        clean_text = re.sub(r'[^a-zA-Z0-9 ]', '', clean_text).lower()
        return clean_text

    def clean(self):
        """Applies text cleaning to the 'sentence' column."""
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_csv()` first.")
        if self.text_col not in self.df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in DataFrame.")
        self.df[self.text_col] = self.df[self.text_col].apply(self._clean_string)
        return self.df

    def reduce_to_columns(self):
        """
        Reduces the DataFrame to only the text and label columns.
        Useful for ensuring the model only processes the necessary columns.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_csv()` first.")
        if self.df_cache is None:
            self.df_cache = self.df
        self.df = self.df[[self.text_col, self.label_col]].copy()

        return self.df
    

    # Create a method that transforms the labels to binary, hvile keeping the original labels in continous_labl column:
    def transform_labels_to_binary(self, positive_threshold =0.5, negative_threshold=0.5):
        """
        Transforms labels to binary ('positive', 'negative') while preserving original labels in 'continuous_label'.

        Either provide:
        - `positive_labels` and `negative_labels` as sets/lists of discrete labels
        OR
        - `threshold` for numeric labels (everything above is positive, below is negative)

        Labels not matching either positive or negative categories will be labeled as 'neutral'.

        :param positive_labels: list or set of labels considered positive (used if threshold is None)
        :param negative_labels: list or set of labels considered negative (used if threshold is None)
        :param threshold: numeric value for splitting continuous label values into positive/negative
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_csv()` first.")

        self.df['continuous_label'] = self.df[self.label_col]

            # Continuous label mode
        self.df[self.label_col] = self.df[self.label_col].apply(
            lambda x: "positive" if x >= positive_threshold  else ("negative" if x <= negative_threshold else "neutral")
        )
        return self.df

    def drop_neutral(self):
        """
        Drops rows labeled as 'neutral' in the label column.

        :return: DataFrame
            The updated DataFrame without 'neutral' labels.
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        if self.df_cache is None:
            self.df_cache = self.df        
        self.df = self.df[self.df[self.label_col] != "neutral"].reset_index(drop=True)
        return self.df

    @property
    def texts(self):
        if self.df is None:
            raise ValueError("Data not loaded.")
        return self.df[self.text_col].tolist()

    @property
    def labels(self):
        if self.df is None:
            raise ValueError("Data not loaded.")
        return self.df[self.label_col].tolist()
