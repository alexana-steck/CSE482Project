import requests
import datetime
from datetime import timedelta
import json
from uszipcode import SearchEngine
from noaa_sdk import noaa
from bs4 import BeautifulSoup

WEEKS_OF_FOOTBALL = 16
SCORES_URL = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
TEAMS_URL = "http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"
LOCATIONS_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/locations"
WEATHER_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
ESPN_SCHEDULE = "http://www.espn.com/college-football/schedule"
ESPN_CONFERENCES = "http://www.espn.com/college-football/teams"

YEAR_START = 2002
YEAR_END = 2018

scores = {}
venue_locations = {}

CORE_VALS = {"PRCP" : "Pricipitation", "SNOW" : "Snowfall", "SNWD" : "Snow Depth", "TMAX" : "Maximum Temperature", "TMIN" : "Minimum Temperature", "AWND" : "Average wind speed"}

location_dates_times = {}
weather_data = {}
gamescoreinfo = {}
conferences = {}


n = noaa.NOAA()
search = SearchEngine(simple_zipcode=True)


#Gets venue data in case it doesn't exist in API
for i in range(1,16):
    if i == 1:
        url = ESPN_SCHEDULE
    else:
        url = ESPN_SCHEDULE + "/_/week/" + str(i)

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    all = soup.findAll("table", class_="schedule has-team-logos align-left")

    try:
        for vals in all:
            for item in vals.children:
                for oth in item.children:
                    if oth.attrs and oth.attrs['data-is-neutral-site'] == 'false':
                        try:
                            hometeam = oth.contents[1].contents[0].contents[1].contents[0].contents[0]
                            try:
                                homeloc = oth.contents[5].contents[0].split(",")[1:]
                            except:
                                homeloc = oth.contents[5].contents[1].split(",")[1:]

                            homeloc[0] = homeloc[0].strip()
                            homeloc[1] = homeloc[1].strip()

                            venue_locations[hometeam] = homeloc
                        except:
                            continue
    except:
        continue


response = requests.get(ESPN_CONFERENCES)

soup = BeautifulSoup(response.text, "html.parser")

all = soup.findAll("div", class_="mt7")

for item in all:
    conf = item.contents[0].contents[0]
    for oth in item.contents[1].contents:
        team = oth.contents[0].contents[0].contents[1].contents[0].contents[0].contents[0]
        conferences[team] = conf

#gets the scores and weather data
for year in range(YEAR_START, YEAR_END+1):

    date = datetime.date(year, 8, 27)
    weekday = date.weekday()
    datestr = date.strftime("%Y%m%d")
    r = requests.get(SCORES_URL, params={'dates': datestr})
    first_date = r.json()['leagues'][0]['calendar'][0]['startDate'][:10]
    date = datetime.date(int(first_date[:4]), int(first_date[5:7]), int(first_date[8:]))


    scores[year] = {}
    gamescoreinfo[year] = {}

    for week in range(1, WEEKS_OF_FOOTBALL):
        first = True
        for i in range(2):
            datestr = date.strftime("%Y%m%d")
            r = requests.get(SCORES_URL,
                         params={'dates': datestr, 'limit': 10000})

            if first:
                try:
                    first = False
                    gamescoreinfo[year][week] = {}
                    scores[year][week] = r.json()['events']
                    date += timedelta(1)
                except:
                    continue
            else:
                try:
                    scores[year][week].extend(r.json()['events'])
                except:
                    continue


        try:
            for game in scores[year][week]:
                if game['status']['type']['state'] != 'post':
                    continue

                if 'venue' in game['competitions'][0]:
                    city = game['competitions'][0]['venue']['address']['city']
                    state = game['competitions'][0]['venue']['address']['state']
                elif 'competitors' in game['competitions'][0] and len(game['competitions'][0]['competitors']) == 2:
                    for item in game['competitions'][0]['competitors']:
                        if item['homeAway'] == 'home':
                            loc = item['team']['location']
                            if loc in venue_locations:
                                city = venue_locations[loc][0]
                                state = venue_locations[loc][1]
                else:
                    continue

                try:

                    zipcode = search.by_city_and_state(city, state)
                    zip = "ZIP:" + str(zipcode[0].to_dict()['zipcode'])
                except KeyError:
                    continue
                except IndexError:
                    continue
                citystate = city+", "+state
                gamedatetime = game['competitions'][0]['date']

                gamedate = gamedatetime[:10]
                gametimestart = gamedatetime[-6:-1]
                gametimeend = str(int(gametimestart[:2]) + 4) + gametimestart[2:]

                if state not in location_dates_times:
                    location_dates_times[state] = {}
                if city not in location_dates_times[state]:
                    location_dates_times[state][city] = []

                location_dates_times[state][city].append((gamedate, gametimestart))



                data = requests.get(WEATHER_URL,
                                    params={'datasetid': 'GHCND', 'startdate':gamedate, 'enddate':gamedate, 'locationid' : zip, 'metric': 'standard'},
                                    headers={'token': 'HcyHMmwyZPLKwWCGnpLiJCLjiHZvouHT'}).json()


                if state not in weather_data:
                    weather_data[state] = {}
                if city not in weather_data[state]:
                    weather_data[state][city] = {}

                weather_data[state][city][gamedate] = {}

                if data and data['results']:
                    for item in data['results']:
                        if item['datatype'] in CORE_VALS:
                            if item['datatype'] == "TMAX" or item['datatype'] == "TMIN":
                                weather_data[state][city][gamedate][CORE_VALS[item['datatype']]] = ((item['value']/10)*(9/5))+32
                            else:
                                weather_data[state][city][gamedate][CORE_VALS[item['datatype']]] = item['value']




            date += timedelta(6)
        except KeyError:
            continue


file = open("ScoresData.txt", 'w')
json.dump(scores, file)
file.close()

file1 = open("LocationData.txt", 'w')
json.dump(location_dates_times, file1)
file1.close()

file2 = open("WeatherData.txt", 'w')
json.dump(weather_data, file2)
file2.close()

file3 = open("TeamLoc.txt", 'w')
json.dump(venue_locations, file3)
file3.close()

file4 = open("Conferences.txt", 'w')
json.dump(conferences, file4)
file4.close()