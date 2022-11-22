# Room Occupancy Estimation

**Goal**: estimating the precise number of occupants in a room using multiple non-intrusive environmental sensors like temperature, light, sound, CO2 and PIR.

**Dataset source:**  [Room Occupancy Estimation Data Set](https://archive.ics.uci.edu/ml/datasets/Room+Occupancy+Estimation)


### Data Set Information

The experimental testbed for occupancy estimation was deployed in a 6m x 4.6m room. The setup consisted of 7 sensor nodes and one edge node in a star configuration with the sensor nodes transmitting data to the edge every 30s using wireless transceivers. No HVAC systems were in use while the dataset was being collected.

Five different types of non-intrusive sensors were used in this experiment: temperature, light, sound, CO2 and digital passive infrared (PIR). The CO2, sound and PIR sensors needed manual calibration. For the CO2 sensor, zero-point calibration was manually done before its first use by keeping it in a clean environment for over 20 minutes and then pulling the calibration pin (HD pin) low for over 7s. The sound sensor is essentially a microphone with a variable-gain analog amplifier attached to it. Therefore, the output of this sensor is analog which is read by the microcontroller's ADC in volts. The potentiometer tied to the gain of the amplifier was adjusted to ensure the highest sensitivity. The PIR sensor has two trimpots: one to tweak the sensitivity and the other to tweak the time for which the output stays high after detecting motion. Both of these were adjusted to the highest values. Sensor nodes S1-S4 consisted of temperature, light and sound sensors, S5 had a CO2 sensor and S6 and S7 had one PIR sensor each that were deployed on the ceiling ledges at an angle that maximized the sensor's field of view for motion detection.

The data was collected for a period of 4 days in a controlled manner with the occupancy in the room varying between 0 and 3 people. The ground truth of the occupancy count in the room was noted manually.


## Experimentation

### Data points

The data points within dataset do not make up a continuous time period. The continuous periods are:

2017/12/22 10:49:41 - 2017/12/24 09:10:50
2017/12/25 09:12:12 - 2017/12/26 09:10:01
2018/01/10 15:25:48 - 2018/01/11 09:00:09

Therefore traditional time series analysis is problematic or, at least, needs handling of non-adjacent periods.


### Features

**Temperature**

- Is autocorrelated.
- Expected to be mainly influenced by external factors (outside temperature, room heating) and not by occupancy.

**Light**

- Is autocorrelated.
- Expected to be influenced by external factors (outside natural light) as well as occupancy (eg. lamp is turned on)
- May result in false positives (lamp is not turned off but people left the room)

**Sound**

- Is autocorrelated.
- Expected to be mainly influenced by occupancy therefore can be a good predictor.
- Needs smoothing (moving average)

**CO2**

- Is autocorrelated.
- Expected to be mainly influenced by occupancy but has a large inertia.

**PIR**

- Not autocorrelated.
- Exclusively influenced by occupancy
- Needs smoothing (gives signal only if somebody is moving)


### Experiments

As an experimentation process I plan to tackle the following prediction problems:

1. Binary classification (occupied / not occupied); treat data points being independent
1. Regression (estimate number of people in room); treat data points being independent
1. Binary classification (occupied / not occupied); treat data points as time series
1. Regression (estimate number of people in room); treat data points as time series