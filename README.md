# PyStockAI
Automated system for stock price prediction. Newbie in progress.


### Attributions  
 * [Repo organization](https://docs.python-guide.org/writing/structure/)  
 * [Matplotlib](https://matplotlib.org/)  
 * [Tensorflow](https://www.tensorflow.org/)  
 * [Initial code for using (Tensorflow) to predict stock prices](https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras)  
 * High cost of living  
 * Curiosity -- can I use this for bioinformatic investigation or monitoring data trends in the workplace?

### Results to date (2021-04-10)
 * As expected, the program yields a poor estimation of future stock performance
 * I have moved the code from the tutorial to a single script (predict-train-test.py) and begun modifying it as an exercise in improving my understanding of Python3 code
 * Argparse variables added -- can choose date range for analysis, set tf variables, and of course stock ticker
 * Began trying to improve the visualization -- added seaborn, shaded band representing high/low over time
 * High/low over time does not behave as expected -- upon closer inspection, the "actual" performace over time also appears incorrect
 * Added -k argument to retain dataframe for a run as .csv to allow to debug the representation problem

### Goals
 * Fix the visualization problem
 * Add more useful visuals (volume, etc), change high/low band to predicted high/low band
 * Add module to scrape for additional useful variables to include in model
 * Example scraper module might incorporate weather data -- want scraped data hosted locally so database is merely updated rather than completely rebuilt with each call

### Example output (2021-04-10)
![image](https://user-images.githubusercontent.com/47641830/114276596-1ccd3f00-99dc-11eb-96c5-f29d634a1277.png)

