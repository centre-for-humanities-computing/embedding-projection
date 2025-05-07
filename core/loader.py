import re
import pandas as pd

class CorpusLoader:
    """
    Loads, cleans, and formats text-label datasets from CSV files.
    Standardizes columns to ['sentence', 'label'] for embedding and projection.
    """
    def __init__(self, path, text_col="sentence", label_col="label"):
        self.path = path
        self.text_col = text_col
        self.label_col = label_col
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
        self.df_cache = self.df
        self.df = self.df[[self.text_col, self.label_col]].copy()

        return self.df
    
    def filter_labels(self, positive_labels, negative_labels, relabel=True):
        """
        Creates a new CorpusLoader instance with the DataFrame filtered to only include
        specified positive and negative labels. Optionally relabels them as positive and negative.

        :param positive_labels: list or set of labels considered positive.
        :param negative_labels: list or set of labels considered negative.
        :param relabel: bool
            Whether to convert labels to binary ("positive", "negative").
        :return: CorpusLoader
            A new CorpusLoader instance with filtered data.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_csv()` first.")

        allowed_labels = set(positive_labels) | set(negative_labels)
        filtered_df = self.df[self.df[self.label_col].isin(allowed_labels)].copy()

        if relabel:
            filtered_df[self.label_col] = filtered_df[self.label_col].apply(
                lambda x: "positive" if x in positive_labels else "negative"
            )

        # Create a new instance and assign the filtered DataFrame directly
        new_loader = CorpusLoader(self.path, self.text_col, self.label_col)
        new_loader.df = filtered_df.reset_index(drop=True)
        return new_loader
    



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
