import psycopg2


conn_string = "host='localhost' dbname='indoor_position' user='postgres' password='tiancai' port='5432'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''SELECT DISTINCT(id), spec_id FROM penn_station.image_lookup_1000ms'''
cur.execute(query)
results = cur.fetchall()
cur.close()
conn.commit()
lookup = {}
for item in results:
	lookup[item[1]] = item[0]
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query = '''SELECT image_name, id, spec_id, pred0, pred1, pred2, pred3, pred4, prob0, prob1, prob2, prob3
	FROM penn_station.image_lookup_1000ms a'''
cur.execute(query)
result = cur.fetchall()
cur.close()
conn.commit()

