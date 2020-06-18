
# from bs4 import BeautifulSoup
# import requests
# import sys 


# result = requests.get("https://www.metrolyrics.com/kanye-west-albums-list.html")

# src = result.content
# import re

# soup = BeautifulSoup(src, 'lxml')
# songs = soup.find("div", class_="listbox")
# songs = songs.text.lstrip().rstrip()


# songs = re.sub('[^a-zA-Z0-9 . \- \' \n]','', songs)

# for i in range(5):
# 	songs = songs.replace('\n\n','\n')
# songs = songs.lower()
# songs = songs.replace(" ", "-")
# songs = songs.replace("'", "-")
# songs = songs.replace('.',"-")
# songs = songs.replace('--', '-')
# songs = songs.replace('4-', '4')
# songs = songs.replace('3-', '3')
# songs = songs.split("\n")

# songs_list = []
# for i in range(2,len(songs),2):
# 	songs_list.append(songs[i])
# songs_list = list(set(songs_list))
# songs_list.sort()


# print(len(songs_list))
# for song in songs_list:
# 	result = requests.get("http://www.songlyrics.com/kendrick-lamar/" + song + "-lyrics/")

# 	src = result.content
# 	soup = BeautifulSoup(src, 'lxml')
# 	lyrics = soup.find("p", class_= 'songLyricsV14')
	
# 	with open('kendrick.txt', 'a') as f:
# 		f.write(lyrics.text)


bad_words = ['[', ']', ')', '(']
with open('kendrick.txt', 'r') as oldfile, open('newfile.txt', 'a') as newfile:
	for line in oldfile:
		if not any(bad_word in line for bad_word in bad_words):
			print(line)
			newfile.write(line)

# output = ""
# print(d)



