{% if versiondata.version == "Unreleased" %}
# Unreleased
{% else %}
# [v{{ versiondata.version }}](https://github.com/pybamm-team/PyBaMM/tree/pybamm-v{{ versiondata.version }}) - {{ versiondata.date }}
{% endif %}

{% for section, _ in sections.items() %}
{% for category, val in definitions.items() if category in sections[section] %}
## {{ val["name"] }}

{% for text, values in sections[section][category].items() %}
{% set lines = text.split("\n") %}
- {{ lines[0] }}{% if values %} ({{ values|join(", ") }}){% endif %}

{% for line in lines[1:] %}
{{ line }}
{% endfor %}
{% endfor %}

{% endfor %}
{% endfor %}
