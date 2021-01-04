from libs import utils  # Load TFT helper functions
import sklearn.preprocessing  # Used for data standardization

import data_formatters.base
import data_formatters.volatility

GenericDataFormatter = data_formatters.volatility.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

# Implement formatting functions
class MpwikFormatter(GenericDataFormatter):
    """Defines and formats data for the traffic dataset.

    This also performs z-score normalization across the entire dataset, hence
    re-uses most of the same functions as volatility.

    Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
    """

    # This defines the types used by each column
    _column_definition = [
        ('index', DataTypes.REAL_VALUED, InputTypes.ID),
        ('dummy_static', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

        ('sec_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),

        ('sum_all', DataTypes.REAL_VALUED, InputTypes.TARGET),

        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('dayofweek', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('dayofyear', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Sport events Wroclaw', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('TV events', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('Holidays', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),

        #       ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

        ('value_S', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('value_N', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('value_E', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pompa1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pompa2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pompa3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pompa4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pompa5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pompa6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('temperature', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Czek_value_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Czek_value_min', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Czek_value_max', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Czek_value_std', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Bys_value_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Bys_value_min', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Bys_value_max', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Bys_value_std', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)
    ]

    def split_data(self, df, valid_boundary=198_388, test_boundary=242_827):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data

        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        train = df.loc[df.index < valid_boundary]
        valid = df.loc[(df.index >= valid_boundary - 7) & (df.index < test_boundary)]
        test = df.loc[df.index >= test_boundary - 7]

        print(train.shape, valid.shape, test.shape)
        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
          df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 6 * 26,  # Total width of the Temporal Fusion Decoder
            'num_encoder_steps': 6 * 25,  # Length of LSTM decoder (ie. # historical inputs)
            'num_epochs': 100,  # Max number of epochs for training
            'early_stopping_patience': 5,  # Early stopping threshold for # iterations with no loss improvement
            'multiprocessing_workers': 5  # Number of multi-processing workers
        }

        return fixed_params
