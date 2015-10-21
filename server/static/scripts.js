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

	// Float format helper function
	function formatFloat(f) {
		f = f * 100
		return f.toFixed(1);
	}

	// -----------------
	// MODEL METRICS
	// -----------------
	// Handle click events for model metrics
	$('body').on('click', '.btn-metric', function() {
		$(this).toggleClass('active');
		var metrics = []
		$('.btn-metric.active').each(function() {
			metrics.push($(this).attr('data-name'));
		});
		$('input[name="metrics"]').val(metrics.join(','))
	}).on('click', 'a#evaluate-metrics', function() {
		$('img#metrics-processing').show();

		var formData = new FormData($(this).closest('form')[0]);
		$.ajax({
			method: 'POST',
			dataType: 'json',
			url: '/model/metrics',
			data: formData,
			cache: false,
			processData: false,
    		contentType: false,
			success: function(response) {
				var context  = {
					bleu1: (response.bleu) ? formatFloat(response.bleu[0]) : null,
					bleu2: (response.bleu) ? formatFloat(response.bleu[1]) : null,
					bleu3: (response.bleu) ? formatFloat(response.bleu[2]) : null,
					bleu4: (response.bleu) ? formatFloat(response.bleu[3]) : null,
					rouge: (response.rouge) ? formatFloat(response.rouge) : null,
					cider: (response.cider) ? formatFloat(response.cider) : null,
					meteor: (response.meteor) ? formatFloat(response.meteor) : null,
				};
				var source   = $("#scores-table-template").html();
				var template = Handlebars.compile(source);
				var html     = template(context);
				$('#scores-table').find('tbody').prepend(html);
				$('img#metrics-processing').hide();
			}
		});
	}).on('click', 'a#save-metric', function() {
		$('img#metrics-saving').show();
		var form = $(this).closest('form');
		var metricsFormData = new FormData(form[0]);
		$.ajax({
			method: 'POST',
			dataType: 'json',
			url: '/model/metrics/save',
			data: metricsFormData,
			processData: false,
    		contentType: false,
			success: function(response) {
				form.replaceWith(response.name);
				$('img#metrics-saving').hide();
			}
		});
	}).on('click', 'a#remove-metric', function() {
		$(this).closest('tr').remove();
	});

	// ----------------
	// CONTEXT VALIDATION
	// ----------------
	// Handle keyup events for context validation
	$('body').keyup(function(event) {
		var valid = null;
		var id    = $('div#context-wrapper').attr('data-id');
		console.log(event.which);

		// keypress: q = 113, ] = 93
		// keyup: q = 81, ] = 221
		if (event.which == 81) {
			valid = '1';
		} else if (event.which == 221) {
			valid = '0';
		}

		// Only respond to correct key presses
		if (valid != null) {
			$.ajax({
				method: 'GET',
				dataType: 'json',
				cache: false,
				url: '/context/validate/' + id + '/' + valid,
				success: function(response) {
					$('#context-title').text(response.title);
					$('#context-wrapper').attr('data-id', response.idx);
					// cache buster
					$('#context-image').attr('src', '/static/images/context_image.jpg?t=' + new Date().getTime());
					$('#context-description').text(response.description);
					$('#context-tags').text(response.tags.join(', '));
				}
			});
		}
	});
});