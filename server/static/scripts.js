$(function() {
	$('table#models-table').on('click', 'a.show-options', function() {
		$(this).siblings('div.options-wrapper').toggle();
	});
});