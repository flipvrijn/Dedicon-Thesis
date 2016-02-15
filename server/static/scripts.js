$(function() {
	var images_directory = '/static/images/';

	// ---------------- Models overview ----------------
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

	// Set current name
	$('td').on('click', 'a.rename-model', function() {
		var model_name = $(this).attr('data-name');

		$('#rename-model-modal').find('input[name="old_name"]').val(model_name);
		$('#rename-model-modal').find('input[name="name"]').val(model_name.split('.npz')[0]);
	});

	// ---------------- Training status ----------------

	// Generating a sparkline from data
	$('span.sparkline').each(function() {
		$(this).sparkline('html', {
			'width': '400px',
			'chartRangeMin': 0,
			'disableHighlight': true,
		});
	});

	// Mouseover samples tabs
	$('body').on('mouseover', '.samples-tab a', function() {
		$(this).tab('show');
	});

	// Handle start training button
	$('div.modal').on('click', 'button#start-training', function() {
		var model_name = $('body').find('input[name="model_name"]').val()
		if (model_name) {
			window.location.replace('/training/start/'+model_name+'.npz');
		}
	});

	// Dynamic form handling
	$('#data-type').hide(); // Default: hide

	$('#model-type-select').change(function() {
		if ($(this).val() == 't_attn') {
			$('#data-type').show();
		} else {
			$('#data-type').hide();
		}
	});

	$('a[data-toggle="tab"]').on('show.bs.tab', function(e) {
		$('input[name="data_type"]').val($(e.target).attr('href').slice(1));
	});

	// ---------------- Test model ----------------

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

		$.ajax({
			method: 'GET',
			dataType: 'json',
			url: '/image/random/0',
		}).done(function(response) {
			handle_image_response(response);
		});
	}).on('click', 'button#btn-with-attention', function() {
		// Waiting text
		var busy_status = $(this).find('span.busy');
		busy_status.show();

		var image_path = $('div#image-wrapper').find('img#resulting-image').attr('src');
		image_path = image_path.split('/');
		var image = image_path[image_path.length -1];

		$.ajax({
			method: 'GET',
			dataType: 'json',
			url: '/image/attention/' + image
		}).done(function(response) {
			busy_status.hide();

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
		autoProcessQueue: false,
		init: function() {
			imageDropzone = this;

			$('body').on('click', 'a#send-query', function() {
				imageDropzone.processQueue();
			});
		},
		success: function(file, response) {
			handle_image_response(response);
		},
		sending: function(file, xhr, formData) {
			var elemContext = $('textarea[name="context"]');
			var context = '';
			if (elemContext) {
				context = elemContext.val();
			}
			formData.append('introspect', 0);
			formData.append('context', context)
		},
		complete: function(file) {
			this.removeFile(file);
			$('textarea[name="context"]').val('')
		}
	});

	// Float format helper function
	function formatFloat(f) {
		f = f * 100
		return f.toFixed(1);
	}

	// ---------------- Metrics ----------------
	// Handle click events for model metrics
	function check_caption_generation_status(name) {
		$.ajax({
			method: 'GET',
			dataType: 'json',
			url: '/captions/generate/' + name + '/status',
			success: function(response) {
				if (response.status == 'pending') {
					setTimeout(function() { check_caption_generation_status(); }, 5000);
				} else {
					success(response);
				}
			}
		})
	}
	$('body').on('click', '.btn-metric', function() {
		$(this).toggleClass('active');
		var metrics = []
		$('.btn-metric.active').each(function() {
			metrics.push($(this).attr('data-name'));
		});
		$('input[name="metrics"]').val(metrics.join(','))
	}).on('click', '.hyp-model', function() {
		var name = $(this).attr('data-name');
		console.log('clicked');

		$('.btn-gen-hyp').attr('disabled', 'disabled');
		$('.btn-gen-hyp').find('img').show();

		$.ajax({
			method: 'GET',
			dataType: 'json',
			url: '/captions/generate/' + name,
			success: function(response) {
				// start polling status of generating hyp

				// once complete, enable button and hide loading img
			}
		});
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

	// ---------------- Context validation ----------------
	// Handle keyup events for context validation
	if ($('div#context-wrapper').length) {
		$('body').keyup(function(event) {
			var valid = null;
			var id    = $('div#context-wrapper').attr('data-id');

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
	}
});