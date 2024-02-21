function uncollapse_children(node, valid_names) {
	const collapseParent = valid_names.includes(node.name);
	if (node.children && node.children.length > 0) {
		let collapse_this_node = false;
		for (let i = 0; i < node.children.length; i++) {
			if (valid_names.includes(node.children[i].name)) {
				x = uncollapse_children(node.children[i], valid_names);
				node.children[i] = x[0];
				collapse_this_node = collapse_this_node || x[1];
			}
		}
		if (collapse_this_node) {
			node.collapsed = false;
		}
	}
	return [node, collapseParent];
}

function toTitleCase(str) {
	return str.replace(
		/\w\S*/g,
		(txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase(),
	);
}

function funmap(echarts, config = {}) {
	const gene_json = config.gene_json || "../data/genes.json";
	const gene_element = config.gene_element || "genes";
	const gene_to_module_json =
		config.gene_to_module || "../data/gene_to_module.json";
	const go_json = config.go_json || "../data/go.json";
	const funmap_json = config.funmap_json || "../data/funmap.json";
	const main_element = config.main_chart || "chart";
	const pie_element = config.pie_chart || "pie_chart";
	let timeout_id = 0;
	fetch(gene_json)
		.then((res) => res.json())
		.then((gene_data) => {
			genelist = document.getElementById(gene_element);
			for (let i = 0; i < gene_data.length; i++) {
				const option = document.createElement("option");
				option.value = gene_data[i];
				genelist.appendChild(option);
			}
		})
		.then(() => {
			fetch(gene_to_module_json)
				.then((res) => res.json())
				.then((gene_to_module) => {
					fetch(go_json)
						.then((res) => res.json())
						.then((go_data) => {
							fetch(funmap_json)
								.then((res) => res.json())
								.then((chartData) => {
									// set label.font weight to bold if node size is less than 50. Recursively for children as well
									//
									function setFontWeight(node) {
										if (node.children && node.children.length > 0) {
											for (let i = 0; i < node.children.length; i++) {
												setFontWeight(node.children[i]);
											}
										}
										if (node.size <= 50) {
											node.label = {
												fontWeight: "bold",
											};
										}
									}
									setFontWeight(chartData);
									const chartOptions = {
										title: {
											text: "FunMap Hierarchal Organization",
											left: "center",
											textStyle: {
												fontSize: 30,
												fontWeight: "bolder",
												color: "#333",
											},
										},
										tooltip: {
											trigger: "item",
											triggerOn: "mousemove",
											confine: true,
											formatter: (params, ticket, callback) => {
												node_go = go_data[params.name];
												let hallmark_string = "";
												if ("hallmarks" in params.data) {
													for (
														let i = 0;
														i < params.data.hallmarks.length;
														i++
													) {
														hallmark_string += toTitleCase(
															`${params.data.hallmarks[i]}, `,
														);
													}
													hallmark_string = hallmark_string.slice(0, -2);
												}

												if (hallmark_string === "") {
													hallmark_string = "No Hallmarks";
												}
												const content = `<h5 style="max-width: 100%; text-align:center; word-break: normal; white-space: normal;">${
													params.name
												}</h5><h6 style="max-width: 100%; text-align:center; word-break: normal; white-space: normal">Cancer Hallmarks: ${hallmark_string}</h6><br><h6 style="text-align:center;">Top GO Terms</h6><table style="width: 100%" class="tooltip_table"><thead><tr><th>GO Category</th><th>GO ID</th><th>Name</th><th>p-Value</th></tr></thead><tbody><tr><td>Biological Process</td><td>${
													node_go.gobp.id
												}</td><td>${node_go.gobp.name}</td><td>${
													node_go.gobp.p === "0"
														? "< 2.20e-16"
														: parseFloat(node_go.gobp.p).toExponential(2)
												}</td></tr><tr><td>Cellular Component</td><td>${
													node_go.gocc.id
												}</td><td>${node_go.gocc.name}</td><td>${
													node_go.gocc.p === "0"
														? "< 2.20e-16"
														: parseFloat(node_go.gocc.p).toExponential(2)
												}</td></tr><tr><td>Molecular Function</td><td>${
													node_go.gomf.id
												}</td><td>${node_go.gomf.name}</td><td>${
													node_go.gomf.p === "0"
														? "< 2.20e-16"
														: parseFloat(node_go.gomf.p).toExponential(2)
												}</td></tr></tbody></table>`;
												const el = document.createElement("div");
												el.innerHTML = content;
												el.style.maxWidth = "100%";
												el.style.textAlign = "center";
												el.style.fontSize = "0.55vw";
												el.style.lineHeight = "0.5vw";
												return el;
											},
											extraCssText: "max-width: 30%;",
											className: "tooltip",
										},
										toolbox: {
											show: true,
											feature: {
												restore: {},
												saveAsImage: {},
											},
										},
										series: [
											{
												type: "tree",
												roam: true,
												name: "FunMap",
												data: [chartData],
												top: "10%",
												bottom: "10%",
												left: "10%",
												right: "10%",
												layout: "radial",
												symbol: (_value, params) => {
													node = params.data;
													if (
														node.children &&
														node.children.length > 0 &&
														params.collapsed
													) {
														return "diamond";
													}
													return "circle";
												},
												symbolSize: (_value, params) => {
													node = params.data;
													return node.name === "L1_M1"
														? 51
														: 0.0273 * node.size + 19.454;
													//: 6.7737 * Math.log(node.size) - 1.99;
												},
												initialTreeDepth: 1,
												label: {
													position: "top",
													verticalAlign: "middle",
													align: "right",
													fontSize: 12,
												},
												leaves: {
													label: {
														position: "bottom",
														verticalAlign: "middle",
														align: "left",
													},
												},
												animationDuration: 550,
												animationDurationUpdate: 750,
												lineStyle: {
													width: 7,
													color: "yellow",
												},
												itemStyle: {
													borderWidth: 1,
													color: "yellow",
													borderColor: "black",
												},
												emphasis: {
													focus: "ancestor",
												},
											},
										],
										visualMap: {
											type: "continuous",
											min: -165,
											max: 165,
											seriesIndex: 0,
											title: {
												text: "Signed log10 meta-p",
												left: "center",
												top: "bottom",
											},
											text: [
												"Signed log10 meta-p\n\nHigher in Tumor vs. Normal",
												"Lower in Tumor vs. Normal",
											],
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
									};

									// Initialize the chart
									const chart = echarts.init(
										document.getElementById(main_element),
									);
									chart.setOption(chartOptions);
									window.addEventListener("resize", () => {
										chart.resize();
									});

									const pieChartDom = document.getElementById(pie_element);
									const pie_chart = echarts.init(pieChartDom);
									const option = {
										title: {
											text: "Hallmark Key",
											left: "center",
										},
										series: [
											{
												animationDuration: 0,
												animationDurationUpdate: 0,
												name: "Hallmark Key",
												type: "pie",
												radius: ["40%", "70%"],
												avoidLabelOverlap: false,
												itemStyle: {
													borderRadius: 10,
													borderColor: "#fff",
													borderWidth: 2,
												},
												label: {
													show: false,
													position: "center",
												},
												emphasis: {
													animationDuration: 50,
													label: {
														show: true,
														width: 200,
														fontSize: 18,
														fontWeight: "bold",
														overflow: "break",
													},
												},
												labelLine: {
													show: false,
												},
												// Think about making this not hard coded
												data: [
													{
														name: "EVADING GROWTH SUPPRESSORS",
														value: 1,
														itemStyle: { color: "rgba(96, 50, 2, 0.55)" },
													},
													{
														name: "AVOIDING IMMUNE DESTRUCTION",
														value: 1,
														itemStyle: { color: "rgba(194, 0, 126, 0.55)" },
													},
													{
														name: "ENABLING REPLICATIVE IMMORTALITY",
														value: 1,
														itemStyle: { color: "rgba(13, 104, 161, 0.55)" },
													},
													{
														name: "TUMOR PROMOTING INFLAMMATION",
														value: 1,
														itemStyle: { color: "rgba(213, 101, 23, 0.55)" },
													},
													{
														name: "ACTIVATING INVASION AND METASTASIS",
														value: 1,
														itemStyle: { color: "rgba(25, 23, 24, 0.55)" },
													},
													{
														name: "INDUCING ANGIOGENESIS",
														value: 1,
														itemStyle: { color: "rgba(232, 35, 40, 0.55)" },
													},
													{
														name: "GENOME INSTABILITY AND MUTATION",
														value: 1,
														itemStyle: { color: "rgba(21, 41, 132, 0.55)" },
													},
													{
														name: "RESISTING CELL DEATH",
														value: 1,
														itemStyle: { color: "rgba(112, 125, 134, 0.55)" },
													},
													{
														name: "DEREGULATING CELLULAR ENERGETICS",
														value: 1,
														itemStyle: { color: "rgba(106, 42, 133, 0.55)" },
													},
													{
														name: "SUSTAINING PROLIFERATIVE SIGNALING",
														value: 1,
														itemStyle: { color: "rgba(21, 143, 72, 0.55)" },
													},
													{
														name: "NONE",
														value: 1,
														itemStyle: { color: "rgba(220, 211, 182, 0.55)" },
													},
												],
											},
										],
									};

									option && pie_chart.setOption(option);

									const dag = create_dag(
										echarts,
										"hierarchal_modules",
										"L2_M2",
										"dag",
										false, // ignore_size
										"../", // host
									);
									let dag_chart = null;
									dag.then((res) => {
										dag_chart = res;
										chart.on("mouseover", (params) => {
											update_dag_chart(params.name);
										});
									});

									update_dag_chart = (clique_id, gene = null) => {
										const res = update_dag(
											dag_chart,
											"hierarchal_modules",
											clique_id,
											"../", // host
										);
										res.then((result) => {
											timeout_id++;
											if (result) {
												dag_chart = result;
												if (!is_open["dag-tab-container"]) {
													dagTabContainer.dispatchEvent(new Event("click"));
												}
												if (gene != null) {
													dag_chart.dispatchAction({
														type: "highlight",
														seriesIndex: 0,
														name: gene,
													});
												}
											} else {
												if (!is_open["dag-tab-container"]) {
													dagTabContainer.dispatchEvent(new Event("click"));
												}
												setTimeout(
													(old_timeout_id) => {
														if (
															is_open["dag-tab-container"] &&
															old_timeout_id === timeout_id
														) {
															dagTabContainer.dispatchEvent(new Event("click"));
														}
													},
													2000,
													timeout_id,
												);
											}
										});
									};

									function resize_pie_chart() {
										pie_chart.resize({
											width: pieChartDom.clientWidth,
											height: pieChartDom.clientHeight,
											animation: {
												duration: 500,
											},
										});
										setTimeout(() => {
											pie_chart.clear();
											pie_chart.setOption(option, true);
										}, 1);
									}
									new ResizeObserver(resize_pie_chart).observe(pieChartDom);
									pie_chart.on("mouseover", (params) => {
										const indices = [];
										for (
											let i = 0;
											i < chartOptions.series[0].data[0].children.length;
											i++
										) {
											if (
												chartOptions.series[0].data[0].children[i]
													.hallmarks[0] === params.name
											) {
												indices.push(
													chartOptions.series[0].data[0].children[i].name,
												);
											} else if (
												params.name === "NONE" &&
												chartOptions.series[0].data[0].children[i].hallmarks
													.length === 0
											) {
												indices.push(
													chartOptions.series[0].data[0].children[i].name,
												);
											}
										}
										chart.dispatchAction({
											type: "highlight",
											seriesIndex: 0,
											name: indices,
										});
									});
									const search_button =
										document.getElementById("search_button");
									search_button.addEventListener("click", (e) => {
										const gene = document.getElementById("gene_search").value;
										if (gene in gene_to_module) {
											const modules = gene_to_module[gene];
											let node = chartOptions.series[0].data[0];
											node = uncollapse_children(node, modules)[0];
											const newChartOptions = chartOptions;
											newChartOptions.series[0].data[0] = node;
											chart.setOption(newChartOptions);
											chart.dispatchAction({
												type: "highlight",
												seriesIndex: 0,
												name: modules,
											});

											const search_msg = document.getElementById("search_msg");
											search_msg.className = "search_info";
											search_msg.innerHTML = `Gene found in modules: ${modules.join(
												", ",
											)}.`;
											update_dag_chart(modules[modules.length - 1], gene);
										} else {
											const search_msg = document.getElementById("search_msg");
											search_msg.className = "search_error";
											search_msg.innerHTML = "Gene not found in any module.";
										}
									});
								});
						});
				});
		});
}
