import requests

users = []


with open('musae_DE_target.csv', 'r', encoding='utf-8') as f:
    # remove headers
    line = f.readline().strip().split(',')
    #line = f.readline().strip().split(',')
    for l in f:
        line = l.strip().split(',')
        user = {}
        user['twitch_id'] = line[0]
        user['days'] = line[1]
        user['mature'] = line[2]
        user['views'] = int(line[3])
        user['partner'] = line[4]
        user['id'] = line[5]
        users.append(user)

token = "3472a05shinqy24l7n1dr2rf2gh9zr"
headers = {
    "Authorization": f"Bearer {token}",
    "Client-Id": "a0txaxberll97ez9q9goowc4d2ku6c"
}

user_url = "https://api.twitch.tv/helix/users"
channel_url = "https://api.twitch.tv/helix/channels"
count = 0
for i in range(5,20):
    start = i* 500
    for user in users[start:start + 500]:
        count += 1
        print(count)
        r = requests.get(user_url, headers=headers, params={"id": user['twitch_id']})
        if len(r.json()['data']) > 0:
            data = r.json()['data'][0]
            user['broadcaster_type'] = data['broadcaster_type']
            user['login'] = data['login']
            user['display_name'] = data['display_name']
            user['created_at'] = data['created_at']
            user['new_view_count'] = int(data['view_count'])
            user['view_growth'] = user['new_view_count'] - user['views']
            cr = requests.get(channel_url, headers=headers, params={"broadcaster_id": user['twitch_id']})
            channel_data = cr.json()['data'][0]
            user['last_game_id'] = channel_data["game_id"]
            user['last_game_name'] = channel_data["game_name"]


    with open(f"users{i}.csv", 'w', encoding='utf-8') as f:
        data = ""
        for key in user.keys():
            data += f"{key},"
        data = data[:-1]
        f.write(data)
        f.write('\n')

        for user in users:
            data = ""
            for value in user.values():
                data += f"{value},"
            data = data[:-1]
            f.write(data)
            f.write('\n')