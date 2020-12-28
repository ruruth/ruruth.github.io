---
published: true
---
_This is part of a coursework of Programming for Business Analytics I took this semester. The programming language is Python._<br>
---------<br>
I want to get name, address, number of bedrooms and price of each apartment which is listed on the first five pages of the website [https://www.apartmenthomeliving.com/new-york-city-ny?page=](https://www.apartmenthomeliving.com/new-york-city-ny?page=) and use DataFrame to represent a table of data with rows and columns. Then, output as a csv file.<br><br>
The difference in url of each page is only at the **page=**<br>
[![url.png]({{site.baseurl}}/images/url.png)]({{site.baseurl}}/images/url.png)<br><br>
On each page, under `<div id=**"property_list"**>`, contains all names, addresses, number of bedrooms and prices of each property of that page I want.<br>
[![property_list.png]({{site.baseurl}}/images/property_list.png)]({{site.baseurl}}/images/property_list.png)<br>
<br>
[![property_link.png]({{site.baseurl}}/images/property_link.png)]({{site.baseurl}}/images/property_link.png)<br>
<br>
[![address.png]({{site.baseurl}}/images/address.png)]({{site.baseurl}}/images/address.png)<br>
<br>
[![bedrooms.png]({{site.baseurl}}/images/bedrooms.png)]({{site.baseurl}}/images/bedrooms.png)<br>
<br>
[![prices.png]({{site.baseurl}}/images/prices.png)]({{site.baseurl}}/images/prices.png)<br>
<br><br>
{% highlight python linenos %}
    import webbrowser
    webbrowser.open_new('https://www.apartmenthomeliving.com/new-york-city-ny?page=')
    # Open the webpage that we are going to scrape.
{% endhighlight %}

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
    
    
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
    # The User-Agent request header is a characteristic string that lets servers and network peers 
    # identify the application, operating system, vendor, and/or version of the requesting user agent.
    # Chrome: inspect --> network --> All --> website address --> Headers --> User-Agent
    
    column_names = ["name", "address", "beds", "price"]
    house = pd.DataFrame(columns = column_names)
    # Create an empty DataFrame with column names.
    
    for i in range (1,5):
    # From page 1 to 4.

        url = "https://www.apartmenthomeliving.com/new-york-city-ny?page="+str(i)
        # By observation, changes the page number in url changes to different page.
    
        page = requests.get(url, headers=headers)
        # To get or retrieve data from the link, and send out the header to the website.
    
        soup = BeautifulSoup(page.content, 'html.parser')
        # Use BeautifulSoup to creates a parse tree.
    
        posts = soup.find(id = "property_list")
        # The value of the id attribute must be unique within the HTML document.
    
        name_tags = posts.select(".property_link")
        name = [nt.get_text() for nt in name_tags]
        # Find all name on page i at one time.
    
        address_tags = posts.select(".address")
        address = [at.get_text() for at in address_tags]
        # Find all address on page i at one time.
    
        beds_tags = posts.select(".bedrooms")
        beds = [pt.get_text() for pt in beds_tags]
        # Find all beds of the page i at one time.

        price_tags = posts.select(".prices")
        price = [pt.get_text() for pt in price_tags]
        # Find all price of the page i at one time..
    
        df = pd.DataFrame({
            "name":name,
            "address":address,
            "beds":beds,
            "price":price
            })
        # Converts lists to DataFrame.
    
        house = house.append(df, ignore_index=True)
        # Append the DataFrame of page i to house
  
    house.to_csv('output.csv', mode='w')
    # Output a csv file. 
{% endhighlight %}
