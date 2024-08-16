function create_dag(
	echarts,
	clique_type,
	clique_id,
	element_id = "dag",
	ignore_size = false,
	host = "",
	scale = 1,
	is_gene = false,
) {
	return fetch(`${host}data/dag/${clique_type}/${clique_id}.json`)
		.then((response) => response.json())
		.then((clique_data) => {
			if (clique_data.nodes.length > 50 && !ignore_size) {
				console.log("Clique too large to render");
				return false;
			}

			function autoFontSize() {
				const width = document.getElementById(element_id).offsetWidth;
				const height = document.getElementById(element_id).offsetHeight;
				// let new_size = Math.round(
				//   Math.sqrt(width * width + height * height) /
				//     (60 + Math.log(clique_data.nodes.length)),
				// );
				let new_size = (13 / 672) * height * scale;
				// let new_size = (26 / 1518) * height + (13 / 1518) * width;
				new_size = Math.min(new_size, 20);
				return new_size;
			}
			function autoSymbolSize() {
				return (58 / 13) * autoFontSize();
			}
			function autoEdgeLength() {
				// let width = document.getElementById(element_id).offsetWidth;
				// let height = document.getElementById(element_id).offsetHeight;
				// let new_size = Math.round(
				//   Math.sqrt(width * width + height * height) /
				//     (0.75 + Math.log(clique_data.nodes.length)),
				// );
				// new_size = Math.min(new_size, 300);
				// return 300;
				// return Math.min(
				//   (75 / 11) * autoFontSize() +
				//     150 * (0.05 * clique_data.nodes.length) -
				//     150,
				//   300,
				// );
				return Math.max(
					(100 / 11) * autoFontSize() +
						100 * (0.05 * clique_data.nodes.length) -
						100,
					100,
				);
			}
			function autoMapDimensions() {
				// return array
				// array[0] = width
				// array[1] = height
				const width = document.getElementById(element_id).offsetWidth;
				const height = document.getElementById(element_id).offsetHeight;
				let map_height = Math.round(
					Math.sqrt(width * width + height * height) /
						(7 + Math.log(clique_data.nodes.length)),
				);
				let map_width = Math.round(
					Math.sqrt(width * width + height * height) /
						(90 + Math.log(clique_data.nodes.length)),
				);
				map_height = Math.min(map_height, 140);
				map_width = Math.min(map_width, 20);
				return [map_width, map_height];
			}

			const chartDom = document.getElementById(element_id);
			const myChart = echarts.init(chartDom);
			const map_dimensions = autoMapDimensions();
			console.log(is_gene);
			const option = {
				title: {
					text: !is_gene ? `Clique ID: ${clique_id.split("C")[1]}` : clique_id,
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
				toolbox: {
					show: true,
					feature: {
						restore: {},
						dataView: {
							show: true,
							title: "Data View",
							readOnly: true,
							optionToContent: (opt) => {
								const series = opt.series;
								// var table = '<table style="width:100%;text-align:center"><tbody><tr>'
								//   + '<td>Gene Symbol</td>'
								//   + '<td>Signed P Value</td>'
								//   + '</tr>';
								let table =
									'<textarea style="height: 100%; width: 100%" readonly>Gene Symbol\tSigned P Value\n';
								for (let i = 0, l = series[0].nodes.length; i < l; i++) {
									table += `${series[0].nodes[i].name}\t${series[0].nodes[i].value}\n`;
									// table += '<tr>'
									//   + '<td>' + series[0].nodes[i].name + '</td>'
									//   + '<td>' + series[0].nodes[i].value + '</td>'
									//   + '</tr>';
								}
								table += "</textarea>";
								// table += '</tbody></table>';
								return table;
							},
						},
						saveAsImage: {
							show: true,
							title: "Save as Image",
							type: "png",
						},
					},
				},
				visualMap: {
					type: "continuous",
					min: -165,
					max: 165,
					seriesIndex: 0,
					itemWidth: map_dimensions[0],
					itemHeight: map_dimensions[1],
					title: {
						text: "Signed log10 meta-p",
						left: "center",
						top: "bottom",
					},
					text: ["Tumor\nOver-expressed", "Tumor\nUnder-expressed"],
					precision: 3,
					inRange: {
						color: [
							"#67001f",
							"#b2182b",
							"#d6604d",
							"#f4a582",
							"#fddbc7",
							"#f7f7f7",
							"#d1e5f0",
							"#92c5de",
							"#4393c3",
							"#2166ac",
							"#053061",
						].reverse(),
					},
				},
				series: [
					{
						type: "graph",
						name: `Clique ${clique_id}`,
						draggable: clique_data.nodes.length < 100,
						layout: "force",
						nodes: clique_data.nodes,
						links: clique_data.edges,
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
						},
						itemStyle: {
							borderColor: "#000",
							borderWidth: 1,
						},
						lineStyle: {
							width: clique_data.edges.length < 100 ? 1 : 0.5,
						},
						emphasis: {
							focus: "none",
							label: {
								borderColor: "#000",
								borderWidth: 2,
								backgroundColor: "#FFA505",
								color: "#000",
							},
						},
					},
				],
			};
			myChart.dispatchAction({ type: "restore" });
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
		});
}

function update_dag(chart, clique_type, clique_id, host = "") {
	return fetch(`${host}data/dag/${clique_type}/${clique_id}.json`)
		.then((response) => response.json())
		.then((clique_data) => {
			chart.dispatchAction({ type: "restore" });
			if (clique_data.nodes.length > 50) {
				// chart with text "Too many nodes to display."
				chart.setOption({
					// vertically center title
					title: {
						text: `${clique_id}\nToo many nodes to display.`,
						top: "middle",
					},
					series: [
						{
							nodes: [],
							links: [],
						},
					],
				});
				return false;
			}
			chart.setOption({
				toolbox: {
					show: true,
					feature: {
						restore: {},
					},
				},
				title: {
					text: clique_id,
					top: "top",
				},
				series: [
					{
						name: clique_id,
						nodes: clique_data.nodes,
						links: clique_data.edges,
					},
				],
			});
			return chart;
		});
}
