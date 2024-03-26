from datetime import datetime


def transform_datetime(date_time: str) -> datetime:
    """
    Transform datetime from string to datetime object.
    """

    # Transforming the datetime from string to datetime object
    for format in (
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            date_time = datetime.strptime(date_time, format)
            break
        except ValueError:
            continue

    if type(date_time) != datetime:
        raise ValueError(
            "Could not convert the string into datetime datatype, "
            "please check the format."
        )

    date_time = date_time.replace(microsecond=0, tzinfo=None)

    return date_time
