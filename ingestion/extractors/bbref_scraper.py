import requests
from bs4 import BeautifulSoup
import pandas as pd

# helper function 
def _scrape(url, season, table_class):
    response = requests.get(url)    
    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', {'id':table_class})
    headers = [th.getText() for th in table.find('tr').find_all('th') if th.getText()]

    rows = []
    for row in table.find_all('tr', {'class': 'full_table'}):
        player_cell = row.find(attrs={"data-stat": "player"})
        
        if player_cell:
            player_anchor = player_cell.find('a')
            player_name = player_anchor.getText().strip() if player_anchor else player_cell.getText().strip()
            
            stat_cells = [td.getText().strip() for td in row.find_all('td') if td.get('data-stat') != 'player']

            rows.append([player_name] + stat_cells)
    
    df = pd.DataFrame(rows, columns=headers)
    df['Season'] = season

    return df


def scrape_basic_data(season):
    url = f"https://www.basketball-reference.com/wnba/years/{season}_totals.html" 
    return _scrape(url, season, 'totals')

def scrape_advanced_data(season):
    url = f"https://www.basketball-reference.com/wnba/years/{season}_advanced.html"
    return _scrape(url, season, 'advanced')
