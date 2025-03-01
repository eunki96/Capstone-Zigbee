"""Constants for lightme moment sensor."""

from homeassistant.const import Platform

DOMAIN = "lightme"
BRAND = "Team Zigbee"
MODEL = "Moment Sensor"
SW_VERSION = "1.0.0"

PLATFORMS: list[Platform] = [Platform.SENSOR]

CONF_HOST = "host"
CONF_PORT = "port"
