$(document).ready(function () {
	
	map.on('draw:created', function (e) {

		window.type = e.layerType;
		window.layer = e.layer;
        // When a user finishes editing a shape we get that information here
        // editableLayers.addLayer(layer);
        console.log('draw:created->');
        console.log(JSON.stringify(layer.toGeoJSON()));
    });

	$('#upload').click(function(){

		var path = $("path").attr("d");
		var points = path.match(/(\d+)/g);
		console.log(points);
		var polyCoordinates = [];

		// for (var i = 0; i < points.length; i += 2) {
		// 	var longitude = toLongitude(parseInt(points[i])),//I added the 6 and the following 5 to offset the parent transform
		// 	latitude = toLatitude(parseInt(points[i + 1]));
		// 	polyCoordinates.push(new google.maps.LatLng(latitude, longitude));
		// }
		
		var name = $("#textblank").val();
		$.post("/uploadGeoJSON", {geoJSON: JSON.stringify(layer.toGeoJSON()), description: name} , function(data){alert("success"); $('#textblank').val("");});

	});


});
