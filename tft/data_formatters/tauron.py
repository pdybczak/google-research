import sklearn.preprocessing  # Used for data standardization

import data_formatters.base
import data_formatters.volatility

GenericDataFormatter = data_formatters.volatility.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

# Implement formatting functions
class TauronFormatter(GenericDataFormatter):
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
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('dummy_static', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

        ('Timestamp', DataTypes.DATE, InputTypes.TIME),

        #       ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

        ('10HHE24DS002:pos', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24CW001F:av', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24AF201B:cur', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24DS002:spa', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24DS002:me', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24AF201BXQ51:av', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24DS002:con', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24CT102XQ50:av', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

        ('10HHE24CT101XQ50:av', DataTypes.REAL_VALUED, InputTypes.TARGET),

        ('10HHE24CT001XQ50:av', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('10HHE24CW001XQ50:av', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

        ('10HHE24DS002:ma', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('10HHE24AF201B:ins', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        #       ('10HHE24AF201B:s', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('10HHE24CW001S:av', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('10HHE24AF302:s', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('10HHE24AF302:ins', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),

        ('hour', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
    ]

    def split_data(self, df, valid_boundary=400000, test_boundary=600000):
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
        valid = df.loc[(df.index >= valid_boundary) & (df.index < test_boundary)]
        test = df.loc[df.index >= test_boundary]

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
            'total_time_steps': 6 * 25 + 10,  # Total width of the Temporal Fusion Decoder
            'num_encoder_steps': 6 * 25,  # Length of LSTM decoder (ie. # historical inputs)
            'num_epochs': 100,  # Max number of epochs for training
            'early_stopping_patience': 5,  # Early stopping threshold for # iterations with no loss improvement
            'multiprocessing_workers': 5  # Number of multi-processing workers
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': 320,
            'learning_rate': 0.001,
            'minibatch_size': 128,
            'max_gradient_norm': 100.,
            'num_heads': 4,
            'stack_size': 1
        }

        return model_params
