# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class UsAccidentsMarch23(models.Model):
    id = models.TextField(db_column='ID', blank=True, primary_key=True)  # Field name made lowercase.
    source = models.TextField(db_column='Source', blank=True, null=True)  # Field name made lowercase.
    severity = models.IntegerField(db_column='Severity', blank=True, null=True)  # Field name made lowercase.
    start_time = models.TextField(db_column='Start_Time', blank=True, null=True)  # Field name made lowercase.
    end_time = models.TextField(db_column='End_Time', blank=True, null=True)  # Field name made lowercase.
    start_lat = models.FloatField(db_column='Start_Lat', blank=True, null=True)  # Field name made lowercase.
    start_lng = models.FloatField(db_column='Start_Lng', blank=True, null=True)  # Field name made lowercase.
    end_lat = models.TextField(db_column='End_Lat', blank=True, null=True)  # Field name made lowercase.
    end_lng = models.TextField(db_column='End_Lng', blank=True, null=True)  # Field name made lowercase.
    distance_mi_field = models.FloatField(db_column='Distance(mi)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    description = models.TextField(db_column='Description', blank=True, null=True)  # Field name made lowercase.
    street = models.TextField(db_column='Street', blank=True, null=True)  # Field name made lowercase.
    city = models.TextField(db_column='City', blank=True, null=True)  # Field name made lowercase.
    county = models.TextField(db_column='County', blank=True, null=True)  # Field name made lowercase.
    state = models.TextField(db_column='State', blank=True, null=True)  # Field name made lowercase.
    zipcode = models.TextField(db_column='Zipcode', blank=True, null=True)  # Field name made lowercase.
    country = models.TextField(db_column='Country', blank=True, null=True)  # Field name made lowercase.
    timezone = models.TextField(db_column='Timezone', blank=True, null=True)  # Field name made lowercase.
    airport_code = models.TextField(db_column='Airport_Code', blank=True, null=True)  # Field name made lowercase.
    weather_timestamp = models.TextField(db_column='Weather_Timestamp', blank=True, null=True)  # Field name made lowercase.
    temperature_f_field = models.TextField(db_column='Temperature(F)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    wind_chill_f_field = models.TextField(db_column='Wind_Chill(F)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    humidity_field = models.TextField(db_column='Humidity(%)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    pressure_in_field = models.TextField(db_column='Pressure(in)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    visibility_mi_field = models.TextField(db_column='Visibility(mi)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    wind_direction = models.TextField(db_column='Wind_Direction', blank=True, null=True)  # Field name made lowercase.
    wind_speed_mph_field = models.TextField(db_column='Wind_Speed(mph)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    precipitation_in_field = models.TextField(db_column='Precipitation(in)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    weather_condition = models.TextField(db_column='Weather_Condition', blank=True, null=True)  # Field name made lowercase.
    amenity = models.TextField(db_column='Amenity', blank=True, null=True)  # Field name made lowercase.
    bump = models.TextField(db_column='Bump', blank=True, null=True)  # Field name made lowercase.
    crossing = models.TextField(db_column='Crossing', blank=True, null=True)  # Field name made lowercase.
    give_way = models.TextField(db_column='Give_Way', blank=True, null=True)  # Field name made lowercase.
    junction = models.TextField(db_column='Junction', blank=True, null=True)  # Field name made lowercase.
    no_exit = models.TextField(db_column='No_Exit', blank=True, null=True)  # Field name made lowercase.
    railway = models.TextField(db_column='Railway', blank=True, null=True)  # Field name made lowercase.
    roundabout = models.TextField(db_column='Roundabout', blank=True, null=True)  # Field name made lowercase.
    station = models.TextField(db_column='Station', blank=True, null=True)  # Field name made lowercase.
    stop = models.TextField(db_column='Stop', blank=True, null=True)  # Field name made lowercase.
    traffic_calming = models.TextField(db_column='Traffic_Calming', blank=True, null=True)  # Field name made lowercase.
    traffic_signal = models.TextField(db_column='Traffic_Signal', blank=True, null=True)  # Field name made lowercase.
    turning_loop = models.TextField(db_column='Turning_Loop', blank=True, null=True)  # Field name made lowercase.
    sunrise_sunset = models.TextField(db_column='Sunrise_Sunset', blank=True, null=True)  # Field name made lowercase.
    civil_twilight = models.TextField(db_column='Civil_Twilight', blank=True, null=True)  # Field name made lowercase.
    nautical_twilight = models.TextField(db_column='Nautical_Twilight', blank=True, null=True)  # Field name made lowercase.
    astronomical_twilight = models.TextField(db_column='Astronomical_Twilight', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'us_accidents_march23'


class UsAccidentsTable(models.Model):
    id = models.TextField(db_column='ID', blank=True, primary_key=True)  # Field name made lowercase.
    source = models.TextField(db_column='Source', blank=True, null=True)  # Field name made lowercase.
    severity = models.IntegerField(db_column='Severity', blank=True, null=True)  # Field name made lowercase.
    start_time = models.TextField(db_column='Start_Time', blank=True, null=True)  # Field name made lowercase.
    end_time = models.TextField(db_column='End_Time', blank=True, null=True)  # Field name made lowercase.
    start_lat = models.FloatField(db_column='Start_Lat', blank=True, null=True)  # Field name made lowercase.
    start_lng = models.FloatField(db_column='Start_Lng', blank=True, null=True)  # Field name made lowercase.
    end_lat = models.TextField(db_column='End_Lat', blank=True, null=True)  # Field name made lowercase.
    end_lng = models.TextField(db_column='End_Lng', blank=True, null=True)  # Field name made lowercase.
    distance_mi_field = models.FloatField(db_column='Distance(mi)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    description = models.TextField(db_column='Description', blank=True, null=True)  # Field name made lowercase.
    street = models.TextField(db_column='Street', blank=True, null=True)  # Field name made lowercase.
    city = models.TextField(db_column='City', blank=True, null=True)  # Field name made lowercase.
    county = models.TextField(db_column='County', blank=True, null=True)  # Field name made lowercase.
    state = models.TextField(db_column='State', blank=True, null=True)  # Field name made lowercase.
    zipcode = models.TextField(db_column='Zipcode', blank=True, null=True)  # Field name made lowercase.
    country = models.TextField(db_column='Country', blank=True, null=True)  # Field name made lowercase.
    timezone = models.TextField(db_column='Timezone', blank=True, null=True)  # Field name made lowercase.
    airport_code = models.TextField(db_column='Airport_Code', blank=True, null=True)  # Field name made lowercase.
    weather_timestamp = models.TextField(db_column='Weather_Timestamp', blank=True, null=True)  # Field name made lowercase.
    temperature_f_field = models.FloatField(db_column='Temperature(F)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    wind_chill_f_field = models.TextField(db_column='Wind_Chill(F)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    humidity_field = models.FloatField(db_column='Humidity(%)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    pressure_in_field = models.FloatField(db_column='Pressure(in)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    visibility_mi_field = models.FloatField(db_column='Visibility(mi)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    wind_direction = models.TextField(db_column='Wind_Direction', blank=True, null=True)  # Field name made lowercase.
    wind_speed_mph_field = models.TextField(db_column='Wind_Speed(mph)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    precipitation_in_field = models.TextField(db_column='Precipitation(in)', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    weather_condition = models.TextField(db_column='Weather_Condition', blank=True, null=True)  # Field name made lowercase.
    amenity = models.TextField(db_column='Amenity', blank=True, null=True)  # Field name made lowercase.
    bump = models.TextField(db_column='Bump', blank=True, null=True)  # Field name made lowercase.
    crossing = models.TextField(db_column='Crossing', blank=True, null=True)  # Field name made lowercase.
    give_way = models.TextField(db_column='Give_Way', blank=True, null=True)  # Field name made lowercase.
    junction = models.TextField(db_column='Junction', blank=True, null=True)  # Field name made lowercase.
    no_exit = models.TextField(db_column='No_Exit', blank=True, null=True)  # Field name made lowercase.
    railway = models.TextField(db_column='Railway', blank=True, null=True)  # Field name made lowercase.
    roundabout = models.TextField(db_column='Roundabout', blank=True, null=True)  # Field name made lowercase.
    station = models.TextField(db_column='Station', blank=True, null=True)  # Field name made lowercase.
    stop = models.TextField(db_column='Stop', blank=True, null=True)  # Field name made lowercase.
    traffic_calming = models.TextField(db_column='Traffic_Calming', blank=True, null=True)  # Field name made lowercase.
    traffic_signal = models.TextField(db_column='Traffic_Signal', blank=True, null=True)  # Field name made lowercase.
    turning_loop = models.TextField(db_column='Turning_Loop', blank=True, null=True)  # Field name made lowercase.
    sunrise_sunset = models.TextField(db_column='Sunrise_Sunset', blank=True, null=True)  # Field name made lowercase.
    civil_twilight = models.TextField(db_column='Civil_Twilight', blank=True, null=True)  # Field name made lowercase.
    nautical_twilight = models.TextField(db_column='Nautical_Twilight', blank=True, null=True)  # Field name made lowercase.
    astronomical_twilight = models.TextField(db_column='Astronomical_Twilight', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'us_accidents_table'

class User(models.Model):
    username = models.CharField(max_length=32)
    email=models.EmailField()
    password=models.CharField(max_length=32)