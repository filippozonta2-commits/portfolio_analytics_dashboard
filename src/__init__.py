'''Core analytics package for the Portfolio Analytics Dashboard.'''

import plotly.io as pio


# Keep all existing chart functions compatible while rendering them with the
# dashboard's dark visual language. Several chart helpers explicitly request
# ``plotly_white``; remapping that template here lets the whole application
# switch themes without duplicating layout code across every chart function.
pio.templates['plotly_white'] = pio.templates['plotly_dark']
