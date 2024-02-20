function removeFadeOut(el, speed) {
	const seconds = speed / 1000;
	el.style.transition = `opacity ${seconds} s ease`;
	el.style.opacity = 0;
	setTimeout(() => {
		el.parentNode.removeChild(el);
	}, speed);
}

function set_loading_message(message) {
	const el = document.getElementById("loading-text");
	el.innerHTML = message;
	console.log(message);
}

async function dense(echarts, config = {}) {
	const data_url = config.data_url || "data/dense_modules.json";
	const columns = config.columns || 4;
	const element_id = config.element_id || "dense_tables";
	const classes = config.classes || "square";
	let module_data = {};
	set_loading_message("Downloading dense modules...");
	await fetch(data_url)
		.then((res) => res.json())
		.then((dl_module_data) => {
			module_data = dl_module_data;
		});
	const CLUSTER_COUNT = module_data.length;
	const table = document.getElementById(element_id);
	// create divs with a square layout
	// for each cluster
	let currenttr = document.createElement("tr");
	for (let i = 0; i < CLUSTER_COUNT; i++) {
		if (i % columns === 0 && i !== 0) {
			// add row
			table.appendChild(currenttr);
			currenttr = document.createElement("tr");
		}
		const div = document.createElement("div");
		div.className = classes;
		div.id = `cluster_${i + 1}`;
		currenttr.appendChild(div);
	}
	table.appendChild(currenttr);
	for (let i = 0; i < CLUSTER_COUNT; i++) {
		set_loading_message(
			`Loading ${i + 1} of ${CLUSTER_COUNT} dense modules...`,
		);
		setTimeout(() => {
			create_dense_dag(
				echarts,
				`cluster_${i + 1}`,
				module_data[i].name,
				module_data[i].data.nodes,
				module_data[i].data.edges,
			);
		}, 0);
	}
	set_loading_message("Done!");
	removeFadeOut(document.getElementById("loading"), 3000);
}

function create_dense_dag(echarts, element_id, clique_id, nodes, edges) {
	function autoFontSize() {
		const width = document.getElementById(element_id).offsetWidth;
		const height = document.getElementById(element_id).offsetHeight;
		let new_size = Math.round(
			Math.sqrt(width * width + height * height) /
				(40 + Math.log(nodes.length)),
		);
		new_size = Math.min(new_size, 20);
		return new_size;
	}
	function autoSymbolSize() {
		const width = document.getElementById(element_id).offsetWidth;
		const height = document.getElementById(element_id).offsetHeight;
		let new_size = Math.round(
			Math.sqrt(width * width + height * height) /
				(12 + Math.log(nodes.length)),
		);
		new_size = Math.min(new_size, 60);
		return new_size;
	}
	function autoEdgeLength() {
		const width = document.getElementById(element_id).offsetWidth;
		const height = document.getElementById(element_id).offsetHeight;
		let new_size = Math.round(
			Math.sqrt(width * width + height * height) /
				(0.75 + Math.log(nodes.length)),
		);
		new_size = Math.min(new_size, 300);
		return new_size;
	}

	const chartDom = document.getElementById(element_id);
	const myChart = echarts.init(chartDom);
	const option = {
		title: {
			text: clique_id,
			left: "center",
			textStyle: {
				fontSize: autoFontSize() + 10,
				fontWeight: "bolder",
				color: "#333",
			},
		},
		tooltip: {
			trigger: "item",
			triggerOn: "mousemove",
			formatter: "{b}",
			backgroundColor: "#F6F8FC",
			borderColor: "#8C8D8E",
			borderWidth: 1,
			padding: [3, 3, 3, 3],
			textStyle: {
				color: "#4C5058",
				fontSize: autoFontSize(),
			},
		},
		series: [
			{
				type: "graph",
				name: `Clique ${clique_id}`,
				draggable: false,
				layout: "force",
				nodes: nodes,
				links: edges,
				coordinateSystem: null,
				roam: true,
				symbolSize: autoSymbolSize(),
				label: {
					show: true,
					position: "inside",
					fontSize: autoFontSize(),
					padding: [3, 3, 3, 3],
					verticalAlign: "middle",
					color: "#4C5058",
					backgroundColor: "#F6F8FC",
					borderColor: "#8C8D8E",
					borderWidth: 1,
					borderRadius: 4,
				},
				force: {
					repulsion: 300,
					edgeLength: autoEdgeLength(),
					friction: 0.05,
					initLayout: "circular",
					layoutAnimation: false,
				},
				lineStyle: {
					width: edges.length < 100 ? 1 : 0.5,
				},
				emphasis: {
					focus: "none",
					itemStyle: {
						borderColor: "#000",
						borderWidth: 2,
						color: "#FFA505",
					},
				},
			},
		],
	};
	option && myChart.setOption(option);
	window.addEventListener("resize", () => {
		myChart.resize();
		myChart.setOption({
			series: {
				label: {
					textStyle: {
						fontSize: autoFontSize(),
					},
				},
				symbolSize: autoSymbolSize(),
				force: {
					edgeLength: autoEdgeLength(),
				},
			},
		});
	});
	return myChart;
}

function dense_svg(config = {}) {
	const CLUSTER_COUNT = 281;
	const columns = config.columns || 5;
	const element_id = config.element_id || "dense_tables";
	const classes = config.classes || "square";
	const image_count = config.image_count || CLUSTER_COUNT;

	const table = document.getElementById(element_id);
	// create divs with a square layout
	// for each cluster
	let currenttr = document.createElement("tr");
	const bucket = [];
	if (image_count !== CLUSTER_COUNT) {
		for (let i = 0; i < CLUSTER_COUNT; i++) {
			bucket.push(i);
		}

		function getRandomFromBucket() {
			const randomIndex = Math.floor(Math.random() * bucket.length);
			return bucket.splice(randomIndex, 1)[0];
		}

		const random_indices = [];
		for (let i = 0; i < image_count; i++) {
			random_indices.push(getRandomFromBucket());
		}
		random_indices.sort((a, b) => a - b);
		for (let i = 0; i < image_count; i++) {
			if (i % columns === 0 && i !== 0) {
				// add row
				table.appendChild(currenttr);
				currenttr = document.createElement("tr");
			}
			const img = document.createElement("img");
			const cluster_id = random_indices[i];
			img.className = classes;
			img.id = `cluster_${i}`;
			img.src = `data/svg/C${cluster_id + 1}.svg`;
			currenttr.appendChild(img);
		}
		table.appendChild(currenttr);
		set_loading_message("Done!");
		removeFadeOut(document.getElementById("loading"), 3000);
	} else {
		for (let i = 0; i < CLUSTER_COUNT; i++) {
			if (i % columns === 0 && i !== 0) {
				// add row
				table.appendChild(currenttr);
				currenttr = document.createElement("tr");
			}
			const img = document.createElement("img");
			img.className = classes;
			img.id = `cluster_${i + 1}`;
			img.src = `data/svg/C${i + 1}.svg`;
			currenttr.appendChild(img);
		}
		table.appendChild(currenttr);
		set_loading_message("Done!");
		removeFadeOut(document.getElementById("loading"), 3000);
	}
}
