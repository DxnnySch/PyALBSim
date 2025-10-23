import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Example data (replace with your measurements)
X = np.array([
    [21.16796612623362, 3.604368727827011, 6930340898],
    [18.442469528902997, 9.757460242469344, 1229199731],
    [6.9766676913370755, 4.743569520031599, 5624786084],
    [43.0893282646285, 6.31007731967681, 8763823312],
    [39.96109269806246, 8.147054053107126, 4383292864],
    [26.748294728913656, 7.398973563408624, 9995620577],
    [28.1465497285541, 2.7507156535639967, 4730870068],
    [48.40016216388081, 1.3719346535743053, 3213049963],
    [32.471638397169905, 0.14750269789206083, 2125948559],
    [11.516963111380008, 5.953882933817386, 7359192178],

    [38.523340804013316, 8.93784443593217, 9889034897],
    [23.683980018609734, 0.3668938582862651, 3114260056],
    [48.12390969122104, 4.925287525394701, 1094793148],
    [19.28059635141728, 7.662570493077605, 7034374252],
    [9.522289612353426, 3.019580886770012, 5577601052],

    [34.75892583543083, 0.8392970614633419, 3297045855.0],
    [40.94122625988781, 5.833085066330655, 8481580418.0],
    [14.798043155283302, 8.723405006182846, 6490803646.0],
    [46.68268759378331, 7.28238663544561, 4183432919.0],
    [20.608568530161595, 1.5127258670736086, 5622861911.0],
    [26.066142099457522, 9.582213718684137, 2374311383.0],
    [8.797524294495704, 4.350328974118979, 9102711564.0],
    [31.194940041151845, 3.369618610770794, 7371166558.0],
    [10.179105580101814, 2.639432430369478, 4792614508.0],
    [43.24578525136909, 6.468597793818481, 1562918356.0],
])
y_start = np.array([1000, 155, 260, 2550, 1175, 1825, 900, 1050, 470, 575,    2600, 505, 355, 925, 360,    785, 2375, 650, 1340, 790, 420, 545, 1575, 330, 460])
y_end   = np.array([1100, 170, 310, 2800, 1350, 2000, 1000, 1150, 500, 660,    2900, 530, 420, 1050, 500,    830, 2630, 740, 1450, 880, 465, 840, 1800, 450, 600])

# Fit two separate regressions (for xmin and xmax)
model_start = LinearRegression().fit(X, y_start)
model_end   = LinearRegression().fit(X, y_end)

# Degree-2 polynomial features + linear model
model_start_poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
model_end_poly   = make_pipeline(PolynomialFeatures(2), LinearRegression())

model_start_poly.fit(X, y_start)
model_end_poly.fit(X, y_end)

def predict_xlim(flying_height, water_depth, sample_rate, use_poly=True):
    params = np.array([[flying_height, water_depth, sample_rate]])
    if use_poly:
        xmin = model_start_poly.predict(params)[0]
        xmax = model_end_poly.predict(params)[0]
    else:
        xmin = model_start.predict(params)[0]
        xmax = model_end.predict(params)[0]
    return (xmin, xmax)



if __name__ == "__main__":
    # Check R²
    print("R² start:", model_start.score(X, y_start))
    print("R² end:", model_end.score(X, y_end))

    print("xmin = ", model_start.intercept_, "+", model_start.coef_)
    print("xmax = ", model_end.intercept_, "+", model_end.coef_)
    
    print("\nPolynomial Regression (degree 2):")
    print("  R² start:", model_start_poly.score(X, y_start))
    print("  R² end:  ", model_end_poly.score(X, y_end))
    
    print("polynomial", predict_xlim(40.07, 2.4, 7787875987))
    print("linear", predict_xlim(40.07, 2.4, 7787875987, False))
