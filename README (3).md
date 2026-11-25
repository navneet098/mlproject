# Data directory

The source dataset is not committed. Download it from the official UCI page:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The pipeline downloads `household_power_consumption.txt.zip` from the UCI archive and extracts `household_power_consumption.txt` into `data/raw/`. To use a manually downloaded copy, place the text file at that exact path.

The original dataset contains one-minute measurements. WattWise parses `Date` and `Time`, converts `?` to missing values, coerces measurement columns to numeric, sorts timestamps, removes duplicate timestamps, and resamples `Global_active_power` to hourly mean kW. Missing hourly periods are not silently interpolated; incomplete rows are dropped after feature construction.
