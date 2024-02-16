function clique_page(echarts) {
    function download(json_object, file_name = "clique.json") {
        let dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(json_object));
        let downloadAnchorNode = document.createElement("a");
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", file_name);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }
    function isInt(n) {
        return !isNaN(parseInt(n)) && isFinite(n);
    }

    function isNumeric(n) {
        return !isNaN(parseFloat(n)) && isFinite(n);
    }

    function createTable(data) {
        let table = document.createElement("table");
        let thead = document.createElement("thead");
        let tbody = document.createElement("tbody");
        let header = document.createElement("tr");
        for (let i = 0; i < data["rows"].length; i++) {
            let th = document.createElement("th");
            let val = data["rows"][i];
            th.innerHTML = val;
            header.appendChild(th);
        }
        thead.appendChild(header);
        table.appendChild(thead);
        for (let i = 0; i < data["data"].length; i++) {
            let row = document.createElement("tr");
            for (let j = 0; j < data["data"][i].length; j++) {
                let td = document.createElement("td");
                let val = data["data"][i][j];
                if (isNumeric(val)) {
                    val = parseFloat(val);
                    if (val == 0) {
                        val = "< 2.20e-16";
                    } else if (Math.abs(val) > 0.001 && Math.abs(val) < 1000) {
                        val = val.toPrecision(3);
                    } else {
                        val = val.toExponential(2);
                    }
                } else if (val == null) {
                    val = "Not Available";
                }
                td.innerHTML = val;
                row.appendChild(td);
            }
            tbody.appendChild(row);
        }
        table.appendChild(tbody);
        return table;
    }

    let gene_search_error = document.getElementById("gene_search_error");
    let clique_search_error = document.getElementById("clique_search_error");
    let gene_json = "../data/clique_genes.json";
    let gene_element = "genes";
    let gene_to_clique = "../data/gene_to_clique.json";
    fetch(gene_to_clique)
        .then((res) => res.json())
        .then((gene_data) => {
            gene_to_clique = gene_data;
        });
    fetch(gene_json)
        .then((res) => res.json())
        .then((gene_data) => {
            let genelist = document.getElementById(gene_element);
            for (let i = 0; i < gene_data.length; i++) {
                let option = document.createElement("option");
                option.value = gene_data[i];
                genelist.appendChild(option);
            }
            if (window.location.search) {
                let params = new URLSearchParams(window.location.search);
                let clique_id = params.get("clique_id");
                if (clique_id) {
                    clique_id = "C" + clique_id;
                    document.getElementById("clique_id").value = `Clique ID: ${clique_id.replace("C", "")}`;
                    clique_button.click();
                }
                let gene_id = params.get("gene_id");
                if (gene_id) {
                    document.getElementById("gene_id").value = gene_id;
                    gene_button.click();
                }
                if (!clique_id && !gene_id) create_dags(["C160"]);
            } else {
                create_dags(["C160"]);
            }
        });
    let search_results = document.getElementById("search_results");
    let clique_button = document.getElementById("search_button");
    clique_button.addEventListener("click", function () {
        gene_search_error.innerHTML = "";
        let clique_id = document.getElementById("clique_id").value;
        clique_id = "C" + clique_id.split("Clique ID: ")[1];
        if (!clique_id || !isInt(clique_id.replace("C", ""))) {
            clique_search_error.innerHTML = "Please enter a valid Clique ID between 1 and 281.";
            return;
        }
        if (parseInt(clique_id.replace("C", "")) > 281 || parseInt(clique_id.replace("C", "")) < 1) {
            clique_search_error.innerHTML = "Please enter a valid Clique ID between 1 and 281.";
            return;
        }
        search_results.innerHTML = "";
        clique_search_error.innerHTML = "";
        window.history.pushState({}, "", `?clique_id=${clique_id.replace("C", "")}`);
        create_dags([clique_id]);
    });
    let gene_button = document.getElementById("gene_search_button");
    gene_button.addEventListener("click", function () {
        clique_search_error.innerHTML = "";
        if (!gene_to_clique) {
            gene_search_error.innerHTML = "Gene to clique data not loaded yet. Please try again.";
            return;
        } else if (!gene_to_clique[document.getElementById("gene_id").value]) {
            gene_search_error.innerHTML = "Gene not found in any cliques. Please use the dropdown to select valid genes.";
            return;
        }
        gene_search_error.innerHTML = "";
        let gene_id = document.getElementById("gene_id").value;
        let found_cliques = gene_to_clique[gene_id]
        let clique_form = found_cliques.length > 1 ? "cliques" : "clique";
        search_results.innerHTML = `${gene_id} is a member of ${found_cliques.length} ${clique_form}: ${found_cliques.join(", ").replace(/C/g, "")}`;
        if (gene_to_clique[gene_id]) {
            window.history.pushState({}, "", `?gene_id=${gene_id}`);
            create_dags(found_cliques, gene_id);
        }
    });
    document
        .getElementById("clique_id")
        .addEventListener("keyup", function (event) {
            event.preventDefault();
            if (event.keyCode === 13) {
                document.getElementById("search_button").click();
            }
        });
    document
        .getElementById("gene_id")
        .addEventListener("keyup", function (event) {
            event.preventDefault();
            if (event.keyCode === 13) {
                gene_button.click();
            }
        });
    document.getElementById("clique_id").oninput = function () {
        // add Clique prefix if not present
        let clique_id = document.getElementById("clique_id").value;
        if (!clique_id.startsWith("Clique ID: ")) {
            document.getElementById("clique_id").value = "Clique ID: ";
        }
    };
    document.getElementById("clique_id").focus();

    function create_dags(clique_ids, gene_id = null) {
        let result_container = document.getElementById("result_container");
        result_container.innerHTML = "";
        for (let i = 0; i < clique_ids.length; i++) {
            let head_div = document.createElement("div");
            head_div.id = `head_div_${i}`;
            head_div.className = "result";
            if (i != 0) head_div.appendChild(document.createElement("hr"));
            let clique_title_header = document.createElement("h3");
            clique_title_header.id = `clique_title_${i}`;
            clique_title_header.style.flex = "1";
            clique_title_header.style.height = "min-content";
            clique_title_header.style.margin = "0pt";
            clique_title_header.style.columnGap = "0pt";
            clique_title_header.innerHTML = `Clique ID: ${clique_ids[i].replace("C", "")}`;
            head_div.appendChild(clique_title_header);
            let result_sub = document.createElement("div");
            result_sub.id = `result_sub_${i}`;
            result_sub.className = "result_sub";
            let info_div = document.createElement("div");
            info_div.id = `info_div_${i}`;
            info_div.className = "info_div";
            let table_header = document.createElement("h5");
            table_header.className = "info_heading";
            table_header.innerHTML = "GO Enrichment";
            let subtitle = document.createElement("small");
            subtitle.className = "subtitle";
            subtitle.innerHTML = "Top enriched GO terms for clique using over-representation analysis";
            info_div.appendChild(table_header);
            info_div.appendChild(subtitle);
            let table = document.createElement("table");
            table.style.marginTop = "1em";
            let thead = document.createElement("thead");
            let tr = document.createElement("tr");
            let th1 = document.createElement("th");
            th1.innerHTML = "GO Category";
            let th2 = document.createElement("th");
            th2.innerHTML = "Go ID";
            let th3 = document.createElement("th");
            th3.innerHTML = "Go Term";
            let th4 = document.createElement("th");
            th4.innerHTML = "<i>p</i>-Value";
            th4.style.whiteSpace = "nowrap";
            let th5 = document.createElement("th");
            th5.innerHTML = "FDR";
            tr.appendChild(th1);
            tr.appendChild(th2);
            tr.appendChild(th3);
            tr.appendChild(th4);
            tr.appendChild(th5);
            thead.appendChild(tr);
            let table_body = document.createElement("tbody");
            table_body.id = `table_body_${i}`;
            table.appendChild(thead);
            table.className = "long_table";
            let clique_id = clique_ids[i];
            let clique_type = "dense_modules";
            let dag_container = document.createElement("div");
            dag_container.style.flexGrow = "1";

            let dag_div = document.createElement("div");
            dag_div.id = `dag_${i}`;
            dag_div.className = "dag";
            let dag_p = document.createElement("p");
            dag_p.style.color = "var(--color-body-light)";
            dag_p.style.fontSize = "var(--unit)";
            dag_p.style.textAlign = "center";
            dag_p.innerHTML = "Gene significance across 5 cohorts (signed -log<sub>10</sub> meta-<i>p</i>).<br>Scroll to zoom, click and drag to move nodes.";
            // let download_button = document.createElement("button");
            // download_button.id = `download_button_${i}`;
            // download_button.className = "download_button";
            // download_button.innerHTML = "Download Clique";
            dag_container.appendChild(dag_div);
            dag_container.appendChild(dag_p);
            // dag_container.appendChild(download_button);
            info_div.appendChild(table);
            result_sub.appendChild(info_div);
            result_sub.appendChild(dag_container);
            head_div.appendChild(result_sub);
            result_container.appendChild(head_div);
            fetch(`../data/go/clique/${clique_id}.json`)
                .then((res) => res.json())
                .then((data) => {
                    table_body.innerHTML = "";
                    let categories = ["gobp", "gomf", "gocc"];
                    let category_name = {
                        gobp: "Biological Process",
                        gomf: "Molecular Function",
                        gocc: "Cellular Component",
                    };
                    for (let i = 0; i < categories.length; i++) {
                        let go_cat = categories[i];
                        let row = document.createElement("tr");
                        let go_category = document.createElement("td");
                        let go_id = document.createElement("td");
                        let go_term = document.createElement("td");
                        let p_col = document.createElement("td");
                        let fdr = document.createElement("td");
                        go_category.innerHTML = category_name[go_cat];
                        let go_id_text = data[go_cat].set;
                        go_id.innerHTML = `<a href="https://amigo.geneontology.org/amigo/term/${go_id_text}" target="_blank">${go_id_text}</a>`;
                        go_term.innerHTML = data[go_cat].set_name;
                        let p_val = parseFloat(data[go_cat].p);
                        p_col.innerHTML = p_val != 0.0 ? p_val.toExponential(2) : "< 2.20e-16";
                        let fdr_val = parseFloat(data[go_cat].fdr);
                        fdr.innerHTML = fdr_val != 0.0 ? fdr_val.toExponential(2) : "< 2.20e-16";
                        row.appendChild(go_category);
                        row.appendChild(go_id);
                        row.appendChild(go_term);
                        row.appendChild(p_col);
                        row.appendChild(fdr);
                        table_body.appendChild(row);
                    }
                    table.appendChild(table_body);
                }).then(() => {
                    fetch("../data/table/clique/" + clique_id + ".json")
                        .then((res) => res.json())
                        .then((data) => {
                            //  Tumor expression
                            let tumor_expression_table = createTable(data["tumor_expression"]);
                            tumor_expression_table.style.marginTop = "1em";
                            let tumor_expression_header = document.createElement("h5");
                            tumor_expression_header.className = "info_heading";
                            tumor_expression_header.innerHTML = "Tumor Expression by Cohort";
                            let tumor_expression_subtitle = document.createElement("small");
                            tumor_expression_subtitle.className = "subtitle";
                            tumor_expression_subtitle.innerHTML = "The <i>p</i>-values were derived from difference in average protein abundance of clique genes between tumor and normal samples by Wilcoxon rank-sum test.";
                            info_div.appendChild(tumor_expression_header);
                            info_div.appendChild(tumor_expression_subtitle);
                            info_div.appendChild(tumor_expression_table);
                            // Survival
                            let survival_table = createTable(data["survival"]);
                            survival_table.style.marginTop = "1em";
                            let survival_header = document.createElement("h5");
                            survival_header.className = "info_heading";
                            survival_header.innerHTML = "Patient Overall Survival by Cohort";
                            let survival_subtitle = document.createElement("small");
                            survival_subtitle.className = "subtitle";
                            survival_subtitle.innerHTML = "Logrank <i>p</i>-values were derived from Cox-proportional hazard models using average protein abundance of clique genes stratified by median.";
                            info_div.appendChild(survival_header);
                            info_div.appendChild(survival_subtitle);
                            info_div.appendChild(survival_table);
                        }).then(() => {
                            dag_chart = create_dag(
                                echarts,
                                clique_type,
                                clique_id,
                                dag_div.id,
                                (ignore_size = true),
                                (host = "../"),
                                scale = 1.15
                            );
                            dag_chart.then((chart) => {
                                let member_header = document.createElement("h5");
                                member_header.className = "info_heading";
                                member_header.innerHTML = "Clique Members";
                                let member_subtitle = document.createElement("small");
                                member_subtitle.className = "subtitle";
                                let member_list = document.createElement("div");
                                let member_data = chart.getOption().series[0].nodes;
                                let member_text = "<p>";
                                for (let i = 0; i < member_data.length; i++) {
                                    let member = member_data[i];
                                    member_text += member.name;
                                    if (i != member_data.length - 1) {
                                        member_text += ", ";
                                    }
                                }
                                member_text += "</p>";
                                member_list.innerHTML = member_text;
                                member_subtitle.innerHTML = `${member_data.length} members`;
                                dag_container.appendChild(member_header);
                                dag_container.appendChild(member_subtitle);
                                dag_container.appendChild(member_list);
                                window.addEventListener("resize", function () {
                                    chart.resize();
                                });
                                chart.resize();
                                // download_button.onclick = function () {
                                //     download(chart.getOption().series[0], `clique_${clique_id.replace("C", "")}.json`);
                                // };
                                if (gene_id != null) {
                                    chart.dispatchAction({
                                        type: "highlight",
                                        seriesIndex: 0,
                                        name: gene_id,
                                    });
                                }
                            });
                        });
                });
        }
    }
}