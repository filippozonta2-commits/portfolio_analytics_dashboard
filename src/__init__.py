'''Core analytics package for the Portfolio Analytics Dashboard.'''

import plotly.io as pio


# Existing chart helpers use plotly_white. Remap it once so every chart
# follows the application theme without duplicating layout code.
pio.templates['plotly_white'] = pio.templates['plotly_dark']
