UPDATE penn_station.area_lookup_50ms c
	SET lat_center=b.lat_cen, lon_center=b.lon_cen
	FROM (SELECT avg(a.lat) as lat_cen, avg(a.lon) as lon_cen, spec_id as spec 
		from penn_station.image_lookup_50ms a group by a.spec_id) b where b.spec = c.spec_id;