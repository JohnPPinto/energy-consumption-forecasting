from great_expectations.core import ExpectationConfiguration, ExpectationSuite


def generate_great_expectation_suite() -> ExpectationSuite:
    """
     This functions generates an instance for Expectation Suite 
     using the great expectation library.

     Returns
     -------
     great_expectation.core.ExpectationSuite
        A instance with all the predefined validation configuration for the Dataframe 
        are been added in the suite using the ExpectationConfiguration module.
    """

    energy_suite = ExpectationSuite(
        expectation_suite_name="energy_consumption_forecast_suite"
    )

    # Adding DataFrame features in great expectation config for validation.
    # DataFrame Columns
    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": [
                    "datetime_dk",
                    "municipality_num",
                    "branch",
                    "consumption_kwh",
                ]
            },
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_table_column_count_to_equal", kwargs={"value": 4}
        )
    )

    # DataFrame Column: datetime_dk
    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "datetime_dk"},
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_strftime_format",
            kwargs={"column": "datetime_dk", "strftime_format": "%Y-%m-%d %H:%M:%S"},
        )
    )

    # DataFrame Column: municipality_num
    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "municipality_num"},
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "municipality_num", "type_": "int32"},
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "municipality_num",
                "value_set ": (
                    250,
                    773,
                    766,
                    219,
                    326,
                    860,
                    420,
                    175,
                    820,
                    575,
                    240,
                    849,
                    573,
                    492,
                    665,
                    187,
                    482,
                    223,
                    329,
                    201,
                    410,
                    161,
                    756,
                    101,
                    540,
                    390,
                    813,
                    185,
                    306,
                    336,
                    147,
                    159,
                    450,
                    269,
                    340,
                    787,
                    707,
                    630,
                    710,
                    479,
                    265,
                    825,
                    173,
                    563,
                    360,
                    163,
                    791,
                    270,
                    740,
                    400,
                    217,
                    657,
                    607,
                    440,
                    253,
                    510,
                    151,
                    741,
                    779,
                    751,
                    376,
                    727,
                    350,
                    561,
                    330,
                    167,
                    840,
                    190,
                    480,
                    370,
                    210,
                    430,
                    760,
                    621,
                    316,
                    183,
                    706,
                    671,
                    810,
                    153,
                    580,
                    230,
                    169,
                    461,
                    165,
                    661,
                    155,
                    730,
                    157,
                    615,
                    260,
                    530,
                    746,
                    550,
                    846,
                    259,
                    851,
                    320,
                ),
            },
        )
    )

    # DataFrame Column: branch
    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "branch"},
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "branch", "type_": "int8"},
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={"column": "branch", "value_set ": (1, 2, 3)},
        )
    )

    # DataFrame Column: consumption_kwh
    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "consumption_kwh"},
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "consumption_kwh", "type_": "float64"},
        )
    )

    energy_suite.add_expectation(
        expectation_configuration=ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={"column": "consumption_kwh", "min_value": 0},
        )
    )

    return energy_suite
