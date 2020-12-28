---
published: true
---
_This is part of a coursework of Programming for Business Analytics I took this semester. The programming language is Python._<br>
***<br><br>
**Open the webpage that we are going to scrape.**<br>
{% highlight python linenos %}
import webbrowser
webbrowser.open_new('https://www.apartmenthomeliving.com/new-york-city-ny')
{% endhighlight %}<br>

{% highlight python linenos %}
import requests
# The requests module allows you to send HTTP requests using Python.
# The HTTP request returns a Response Object with all the response data (content, encoding, status, etc).

from bs4 import BeautifulSoup
# Beautiful Soup is a Python library for pulling data out of HTML and XML files.
# It creates a parse tree for parsed pages that can be used to extract data from HTML.

import pandas as pd
# Pandas is built on the Numpy package and its key data structure is called the DataFrame.
# DataFrames allow you to store and manipulate tabular data in rows of observations and columns of variables.
{% endhighlight %}<br>


