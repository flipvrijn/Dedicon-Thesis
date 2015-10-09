$(function() {
	var images_directory = '/static/images/';

	// Click events for the models overview table
	$('table#models-table').on('click', 'a.show-options', function() {
		$(this).siblings('div.options-wrapper').toggle();
	}).on('click', 'a.toggle-model', function() {
		var model_name = $(this).attr('data-name');
		var action 	   = $(this).attr('data-action');
		var new_action = (action == 'start' ? 'stop' : 'start');
		var new_text   = (action == 'start'? 'Stop' : 'Start');

		// Send request to server
		$.ajax({
			method: 'GET',
			dataType: 'json',
			url: '/model/' + action + '/' + model_name,
		}).done(function(response) {
			if ( ! response.status) {
				new_text = 'Fout!'
			}
		});

		// Change state of the link
		$(this).text(new_text)
	});

	$('span.model-status').each(function() {
		var span = $(this)
		$.ajax({
			method: 'GET',
			dataType: 'json',
			url: '/model/status/' + span.attr('data-name'),
		}).done(function(response) {
			span.html(response.status_text);
		});
	});

	// Generating a sparkline from data
	$('span.sparkline').each(function() {
		$(this).sparkline('html', {
			'width': '400px',
			'chartRangeMin': 0,
			'disableHighlight': true,
		});
	});

	function handle_image_response(response) {
		if (response.error) {
			$('img#random-image-processing').hide();
			$('div.thumbnail').hide();
			$('div#error-message').show();
			$('div#image-wrapper').show();

			$('div#error-message').find('p.text-danger').text(error)
		} else {
			$('p#caption').empty();
			$.each(response.caption, function(i, word) {
				var class_name = (response.introspect) ? 'show-alpha-image' : ''
				var template = '<span class="' + class_name + '" data-name="' + i + '_' + response.image + '">';
				template    += word + ' ';
				template    += '</span>';
				$('p#caption').append(template);
			});			
			$('img#resulting-image').attr('src', images_directory + response.image);
			$('div#image-wrapper').show();
			$('img#random-image-processing').hide();
		}
	}

	// Click events for testing a model
	$('body').on('click', 'a#random-image-btn', function() {
		// Waiting icon
		$('img#random-image-processing').show();

		var introspect = ($('input#introspect').prop('checked')) ? 1 : 0

		$.ajax({
			method: 'GET',
			dataType: 'json',
			url: '/image/random/' + introspect,
		}).done(function(response) {
			handle_image_response(response);
		});
	});

	// Show corresponding alpha image when mouseover
	$('p#caption').on('mouseenter', 'span.show-alpha-image', function() {
		$('img#resulting-alpha-image').show();
		$('img#resulting-image').hide();
		$('img#resulting-alpha-image').attr('src', images_directory + $(this).attr('data-name'));
	}).on('mouseleave', function() {
		$('img#resulting-alpha-image').hide();
		$('img#resulting-image').show();
	});

	// Initialize the dropzone element
	Dropzone.autoDiscover = false;
	$("#image-dropzone").dropzone({
		success: function(file, response) {
			handle_image_response(response);
		},
		sending: function(file, xhr, formData) {
			var introspect = ($('input#introspect').prop('checked')) ? 1 : 0
			formData.append('introspect', introspect);
		}
	});
});