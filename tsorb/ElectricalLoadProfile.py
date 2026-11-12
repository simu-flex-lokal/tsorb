from tsorb.utils.InputData import DataExchangeCsv
from tsorb.OccupancyModel import OccupancyModel
from tsorb.LoadModel import AppliancesModel
from tsorb.LoadModel import LightingModel
import numpy as np
import pandas as pd


class ElectricalLoadProfile(object):
    def __init__(
        self,
        data_ex=None,
        residents=2,
        random_app_seed_per_run=True,
        pre_setup=False,
        weather_data=None,
        resolved_load=False,
        get_hot_water=False,
        freq="60min",
        resample_mean=False,
    ):
        """
        Initializes an Electrical LoadProfile Model with the given Data.
        -----------------------------------------------------------------------
        Parameters:
            data_ex: DataExchange Object, optional (default= None)
                An Object to Exchange Data between the different Submodels used
                used to build th Electrical Load
            residents: int, optional (default = 2)
                Number of residents living in the Household
            random_app_per_run: int, optional (default=True)
                If set the setting of the Appliances owned in the household
                will be seeded. (Same "random" distribution for each
                Initilization with this seed)
            pre_setup: string, optional (default=False)
                If set to a directory Path as a string, a pre chosen setup for
                the Appliance distribution in the household will be
            weather_data: pd.DataFrame, optional (default=None)
                A time indexed pandas dataframe containing weather data with
                the GHI as a column.
            resolved_load: boolean, default: False
                If activated the load model does not only return the total
                load, instead it gives the load profile for each appliance.
            freq: str, default: 60min
                Frequency in which the resulting profile shall get returned.
            get_hot_water: bool, optional (default=False)
                If set, an extra hot water profile is created independent
                from the electricity load of the other devices.
            resample_mean: bool, optional (default=False)
                Decides if the time series get averaged or not for resampling.
        -----------------------------------------------------------------------
        """
        # Initalize DataExchange Object to Exchange data between Models
        if data_ex is None:
            self._data_ex = DataExchangeCsv()
        else:
            self._data_ex = data_ex

        self.resolved_load = resolved_load
        self.get_hot_water = get_hot_water
        self.freq = freq
        self.resample_mean = resample_mean

        if residents > 5:
            raise ValueError('maximum number of "residents" is 5')
        # init related OccupancyModel, LightingModel, AppliancesModel
        self.occ_model = OccupancyModel(self._data_ex, residents, four_states=True)
        self.lig_model = LightingModel(self._data_ex, self.occ_model, weather_data)
        self.app_model = AppliancesModel(
            self._data_ex,
            self.occ_model,
            random_app_seed_per_run,
            pre_setup,
            get_hot_water=self.get_hot_water,
        )

        self._simulated = None

    def run(self, year, day_of_week, day_in_year):
        """
        Runs an Electrical Load Profile Object for the given year and type of
        day (weekday or weekend)
        -----------------------------------------------------------------------
        Parameters:
            year: int, required
                Set the year in which the Electrical Profile will take place
            day_of_week: str, required
                String either it is a weekday "wd" or weekend "we"
            day_in_year: int, required
                numerical day in year counted from 01.01.xxxx counted as 1. 
                Is generated automatically when run in full year calculations
        -----------------------------------------------------------------------
        Returns:
            Saves the total electricity consumption in the given Electrical
            Load Profile, the total heat gain and the occupancy behavior
        """
        # Start the Occupancy Model needed by the following Models
        self.occ_model.run(day_of_week)
        # Run LightingModel and fetch total Consumption for one day (1440 mins)
        self.lig_model.run(year, day_of_week, day_in_year)
        # Run AppModel and fetch total Consumption for one day (1440 mins)
        self.app_model.run(year, day_of_week, day_in_year)
        # Combine App+Lig Consumption to the daily total Consumption
        self.total_consumption = (
            self.app_model.total_consumption + self.lig_model.total_consumption
        )

        if self.get_hot_water:
            self.total_hot_water = self.app_model.total_hot_water

        self.total_heat_gain = (
            self.app_model.total_heat_gain + self.lig_model.total_heat_gain
        )

    def set_seed(self, seed):
        """
        Sets the seed for the random number generation
        :param seed:
        """
        np.random.seed(seed)

    def run_for_year(self, year):
        """
        Repetetively starts the run function for an Electrical Load Profile for
        each day in the chosen year. It will automatically start the run for
        each working/weekend day in the different months.
        -----------------------------------------------------------------------
        Parameters:
            year: int, required
                Set the year in which the Calculation should take place
        -----------------------------------------------------------------------
        Returns:
            totalLoad: numpy array
                Numpy array with the yearly electrical energy demand with a 
                minutewise resolution.
        """

        days = pd.date_range(
            start=str(year), end=str(int(year) + 1), freq="D", tz="Europe/Berlin"
        )[:-1]

        result = self._run_days(year, days)
        self._simulated = year
        return result

    def run_for_period(self, start, end):
        """
        Runs the Electrical Load Profile for whole calendar days within one
        year. Weekday/weekend and the seasonal correction follow the calendar.
        -----------------------------------------------------------------------
        Parameters:
            start: str or pandas.Timestamp, required
                First day of the period.
            end: str or pandas.Timestamp, required
                Last day of the period, included.
        -----------------------------------------------------------------------
        Returns:
            totalLoad: numpy array
                Numpy array with the electrical energy demand of the period
                with a minutewise resolution.
        """
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if last < first:
            raise ValueError("The period ends before it starts: {} to {}".format(start, end))
        # The lighting model looks its irradiance up by year and day of year.
        if first.year != last.year:
            raise ValueError("The period has to stay within one year: {} to {}".format(start, end))

        days = pd.date_range(start=first, end=last, freq="D", tz="Europe/Berlin")
        result = self._run_days(first.year, days)
        self._simulated = (first, last)
        return result

    def _run_days(self, year, days):
        """
        Simulates the given consecutive calendar days and stores their series.
        Occupancy and appliance states carry over from one day to the next.
        -----------------------------------------------------------------------
        Parameters:
            year: int, required
                Year the days belong to.
            days: pandas.DatetimeIndex, required
                The days to simulate, localized to Europe/Berlin.
        -----------------------------------------------------------------------
        Returns:
            totalLoad: numpy array, or a pandas.DataFrame per appliance when
                the load is resolved.
        """
        number_of_days = len(days)
        # build an empty numpy array to later save the total_consumption
        self.total_load = np.empty(1440 * number_of_days, dtype=float)
        self.app_heat_gain = np.empty(1440 * number_of_days, dtype=float)
        self.occ_active = np.empty(144 * number_of_days, dtype=float)
        self.occ_not_active = np.empty(144 * number_of_days, dtype=float)
        if self.get_hot_water:
            self.hotWater = np.empty(1440 * number_of_days, dtype=float)

        self.light = np.empty(1440 * number_of_days, dtype=float)  # Test Lighting

        # coefficients for seasonal correction
        z = [
            -2.46333771e-10,
            2.09410267e-07,
            -4.91019666e-05,
            1.99475890e-03,
            1.13989689e00,
        ]

        # coefficients for correction of hot water demand
        cor_hotwater = 0.7

        if self.resolved_load:
            appload = np.empty(
                [len(self.app_model.loads), 1440 * number_of_days], dtype=float
            )
            app_names = [load.get_key for load in self.app_model.loads]

        # Loop through every day
        for ii, day in enumerate(days):
            # Differentiate between weekday or weekendday
            if day.dayofweek in (5, 6):
                day_of_week = "we"
            else:
                day_of_week = "wd"

            # calculate dynamic factor for seasonal correction
            f = sum(coeff * (day.dayofyear ** pot) for pot, coeff in enumerate(z[::-1]))

            # start the run function for the given day
            self.run(year, day_of_week, day_in_year=day.dayofyear)
            # Save the total_consumption in the totalLoad
            self.total_load[1440 * ii : 1440 * (ii + 1)] = self.total_consumption * f
            self.app_heat_gain[1440 * ii : 1440 * (ii + 1)] = self.total_heat_gain * f
            if self.get_hot_water:
                self.hotWater[1440 * ii : 1440 * (ii + 1)] = (
                    self.total_hot_water * cor_hotwater
                )
            self.occ_active[144 * ii : 144 * (ii + 1)] = self.occ_model.occ_activity
            self.occ_not_active[
                144 * ii : 144 * (ii + 1)
            ] = self.occ_model.occ_no_activity

            #
            if self.resolved_load:
                for iii, load in enumerate(self.app_model.loads):
                    appload[iii][1440 * ii : 1440 * (ii + 1)] = load.consumption * f

            self.light[1440 * ii : 1440 * (ii + 1)] = (
                self.lig_model.total_consumption * f
            )

        if self.resolved_load:
            self.load_resolved = pd.DataFrame(
                appload.T,
                index=pd.date_range(
                    start=days[0].tz_localize(None),
                    periods=len(self.total_load),
                    freq="1min",
                ),
                columns=app_names,
            )

            self.load_resolved["LIGHTS"] = self.light

            return self.load_resolved

        else:
            return self.total_load

    def get_rescheduled_profiles(self, year):
        """
        Gets the relevant profiles (total load, appHeatGain, occActive,
        occNotActive), reschedules them to the frequencies and returns them
        as pandas.DataFrame
        """
        if not self._simulated == year:
            self.run_for_year(year)

        # Point-sampled hot water is what the cached yearly profiles of tsib
        # were generated with, so the yearly path has to keep it.
        return self._rescheduled(pd.Timestamp(year, 1, 1), hot_water_mean=False)

    def get_period_profiles(self, start, end):
        """
        Gets the same profiles as `get_rescheduled_profiles` for whole calendar
        days from start to end, included, indexed from the first day. With
        `resample_mean` the hot water is averaged too, which keeps its energy.
        -----------------------------------------------------------------------
        Parameters:
            start: str or pandas.Timestamp, required
                First day of the period.
            end: str or pandas.Timestamp, required
                Last day of the period, included.
        -----------------------------------------------------------------------
        Returns:
            profiles as pandas.DataFrame
        """
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if not self._simulated == (first, last):
            self.run_for_period(first, last)

        return self._rescheduled(first, hot_water_mean=self.resample_mean)

    def _rescheduled(self, start, hot_water_mean):
        """
        Resamples the simulated minute series to the requested frequency.
        -----------------------------------------------------------------------
        Parameters:
            start: pandas.Timestamp, required
                Timestamp of the first simulated minute.
            hot_water_mean: bool, required
                Whether the hot water is averaged rather than point-sampled.
        -----------------------------------------------------------------------
        Returns:
            profiles as pandas.DataFrame
        """
        if self.resolved_load:
            if self.resample_mean:
                return self.load_resolved.resample(self.freq).mean()
            else:
                return self.load_resolved.resample(self.freq).ffill()
        else:
            index_1 = pd.date_range(
                start=start, periods=len(self.total_load), freq="1min"
            )
            index_10 = pd.date_range(
                start=start, periods=len(self.occ_active), freq="10min"
            )

            profiles = pd.DataFrame([])

            if self.resample_mean:
                profiles["Load"] = (
                    pd.Series(self.total_load, index=index_1).resample(self.freq).mean()
                )
                profiles["AppHeatGain"] = (
                    pd.Series(self.app_heat_gain, index=index_1)
                    .resample(self.freq)
                    .mean()
                )
                profiles["OccActive"] = (
                    pd.Series(self.occ_active, index=index_10)
                    .astype(int)
                    .resample(self.freq)
                    .mean()
                )
                profiles["OccNotActive"] = (
                    pd.Series(self.occ_not_active, index=index_10)
                    .astype(int)
                    .resample(self.freq)
                    .mean()
                )
            else:
                profiles["Load"] = (
                    pd.Series(self.total_load, index=index_1).resample(self.freq).ffill()
                )
                profiles["AppHeatGain"] = (
                    pd.Series(self.app_heat_gain, index=index_1).resample(self.freq).ffill()
                )
                profiles["OccActive"] = (
                    pd.Series(self.occ_active, index=index_10)
                    .astype(int)
                    .resample(self.freq)
                    .ffill()
                )
                profiles["OccNotActive"] = (
                    pd.Series(self.occ_not_active, index=index_10)
                    .astype(int)
                    .resample(self.freq)
                    .ffill()
                )

            if self.get_hot_water:
                hot_water = pd.Series(self.hotWater, index=index_1).resample(self.freq)
                profiles["HotWater"] = hot_water.mean() if hot_water_mean else hot_water.ffill()

            return profiles
