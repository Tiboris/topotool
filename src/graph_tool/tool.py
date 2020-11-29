#!/usr/bin/env python3
import os
import click
import sys
import shelve

import jinja2
import networkx as nx
import matplotlib.pyplot as plt


from os import pipe
from collections import Counter, OrderedDict
from graph_tool.graphs import Graph
from copy import Error, deepcopy
from matplotlib.pyplot import colorbar
from jinja2 import Template
from itertools import chain


GRAPH_DATA = "graph_input_data"
GRAPH_OBJECT = "graph_object"
GRAPH_NX = "graph_networkx"
LEVELS = "levels"
PRED = "predecessors"
BACKBONE = "backbone_edges"
EDGES = "edges"
MASTER = "master_node"
SCALING_DEFAULTS = "../src/graph_tool/data"


def load_context(ctx, storage):
    ctx.ensure_object(dict)
    if not ctx.obj:
        try:
            with shelve.open(storage) as db:
                ctx.obj[GRAPH_DATA] = db[GRAPH_DATA]
                ctx.obj[GRAPH_OBJECT] = db[GRAPH_OBJECT]
                ctx.obj[GRAPH_NX] = db[GRAPH_NX]
                ctx.obj[MASTER] = db[MASTER]
                ctx.obj[EDGES] = db[EDGES]
                ctx.obj[LEVELS] = db[LEVELS]
                ctx.obj[PRED] = db[PRED]
                ctx.obj[BACKBONE] = db[BACKBONE]
        except KeyError:
            pass

    return ctx


def save_data(path, data):
    """
    Writes data with file.write() to file specified by path.
    if path is not specified use stdout.
    """
    with open(path, 'w') as data_file:
        data_file.write(data)


def load_jinja_template(path):
    with open(path) as file_:
        template = Template(file_.read())
    return template


def gen_metadata(name, node_os, destination,
                 project, run, job, template,
                 tool_repo, tool_branch):
    metadata = template.render(
        dnsname=name,
        node_os=node_os,
        project=project,
        run=run,
        job=job,
        tool_repo=tool_repo,
        tool_branch=tool_branch
    )
    output_file = os.path.join(destination, f"{name}.yaml")
    save_data(output_file, metadata)


def generate_topo(nodes):
    raise NotImplementedError()


def create_level(level, max_levels, level_width):
    level_base = f"y{level}"
    # if the actual level is first or last return only level base (one server)
    if level == max_levels - 1 or level == 0:
        return [level_base]

    level = []
    # otherwise create level with number of servers equal to max_level_width
    for pos in range(level_width):
        level.append(f"{level_base}x{pos}")

    return level


def create_basic_topo(max_width, max_levels):
    topo = {
        "levels": {},
        "predecessors": {},
        "edges": [],
        "backbone_edges": [],
    }

    predecessors = {}
    for level in range(max_levels):
        topo["levels"][level] = create_level(level, max_levels, max_width)
        pred_idx = 0
        if level:
            for replica in topo["levels"][level]:
                predecessor = topo["levels"][level - 1][pred_idx]
                predecessors[replica] = predecessor

                if level != max_levels - 1:
                    topo["edges"] = topo["edges"] + [(predecessor, replica)]
                else:  # if max level then add all from previous level as pred
                    topo["edges"] = topo["edges"] + [
                        (pred, replica) for pred in topo["levels"][level-1]
                    ]

                if level - 1 != 0 and level != max_levels - 1:
                    pred_idx += 1

    topo["backbone_edges"] = [
        (value, key) for key, value in predecessors.items()
    ]
    topo["predecessors"] = predecessors

    return topo


@click.group(chain=True)
@click.option("--debug", default=False, is_flag=True)
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.option("-o", "--output", default="./graph_out")
@click.pass_context
def graphcli(ctx, storage, output, debug):
    """graphcli tool for topology"""
    ctx = load_context(ctx, storage)


def levels_from_fist_successors(successors):
    return [[successors[0][0]], list(successors[0][1])]


def predecessors_from_first_levels(levels):
    res = {}
    for node in levels[-1]:
        res[node] = levels[0][0]
    # print(res)
    return res


@graphcli.command()
@click.argument("input_file")
@click.option("-m", "--master", default="y0")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.option(
    "-t", "--type", "data_format", default="ipa",
    type=click.Choice(["ipa", "edges"]),
)
@click.pass_context
def load(ctx, input_file, data_format, master, storage):
    """Loads grap data to dictionary"""
    ctx.ensure_object(dict)

    with open(input_file, "r") as f:
        ctx.obj[GRAPH_DATA] = f.read().splitlines()

    data = deepcopy(ctx.obj[GRAPH_DATA])
    graph_data = Graph(data, data_format=data_format)

    # print(graph_data.edge_list)

    ctx.obj[GRAPH_OBJECT] = graph_data

    G = nx.Graph()
    G.name = f"Loaded graph of topology from {input_file}"
    for vertex in graph_data.vertices:
        G.add_node(vertex)

    G.add_edges_from(graph_data.edge_list)

    ctx.obj[GRAPH_NX] = G
    ctx.obj[MASTER] = master

    print("Loading of topology done.")

    with shelve.open(storage) as db:
        for key in ctx.obj:
            db[key] = ctx.obj[key]


@graphcli.command()
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.option(
    "--branches", "-x",
    type=click.IntRange(3, 20),
    help="Number of topology branches <3-20> to be generated (x-axis)."
)
@click.option(
    "--length", "-y",
    type=click.IntRange(3, 20),
    help="Set topology length <3-20> to be generated (y-axis)."
)
@click.option(
    "--nodes", "-n",
    type=click.IntRange(1, 60),
    help="Needed number of topology nodes length <3-60> (server count)."
)
@click.option(
    "--master", "-m", default="y0",
    help="Master node name."
)
@click.pass_context
def generate(ctx, storage, branches, length, nodes, master):
    """Generate a topology graph based on options."""
    topo = {}
    if nodes:
        topo = generate_topo(nodes)
    else:
        if not branches:
            branches = 3
        if not length:
            length = 3
        topo = create_basic_topo(max_width=branches, max_levels=length)

    G = nx.Graph()
    G.add_edges_from(topo["edges"])

    ctx.obj[GRAPH_NX] = G
    ctx.obj[GRAPH_DATA] = None

    # ctx.obj[LEVELS] = topo["levels"]

    # ctx.obj[PRED] = topo["predecessors"]
    ctx.obj[MASTER] = master
    # ctx.obj[BACKBONE] = compatible_backbone_edges(G, master)

    # ctx.obj[EDGES] = topo["edges"]

    with shelve.open(storage) as db:
        for key in ctx.obj:
            # print(key)
            db[key] = ctx.obj[key]


@graphcli.command()
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def draw(ctx, storage):
    """Draw a topology graph as picture."""
    ctx = load_context(ctx, storage)

    try:
        G = ctx.obj[GRAPH_NX]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


def get_missing_segments(edges):
    segments = {}
    nodes = list(chain(*edges))
    most_common_nodes = Counter(nodes).most_common()
    node_iter = iter(most_common_nodes)
    processed = set()
    not_processed = set(edges)
    while not_processed:
        # from time import sleep
        # sleep(3)
        # print("N", not_p,"P", proc, "A", node, "M", most_common_nodes)
        left = set()
        right = set()
        node, _ = next(node_iter)
        for a, b in not_processed:
            if node == a:
                left.add(b)
                processed.add((a, b))
            elif node == b:
                right.add(a)
                processed.add((a, b))
        not_processed -= processed

        if right.union(left):
            segments.update({node: right.union(left)})

    return segments


def print_topology(topology):
    # in general the topology's second level (y1x*) should be there always
    topo_width = len(topology[1])
    for level in topology:
        if len(topology[level]) == 1:
            ws = "".join(["\t" for _ in range(int(topo_width/2))])
            line = f"{ws}{topology[level][0]}"
        else:
            line = "\t".join(topology[level])
        print(line)


@graphcli.command()
@click.option("-d", "--out-dir", default="./FILES")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.option(
    "-j", "--jenkins-template", type=click.Path(exists=True),
    default=os.path.join(
        SCALING_DEFAULTS,
        "jenkinsjob.j2"
    ),
    help="Jenkinsfile jinja2 template", show_default=True
)  # FIXME paths ^^^
@click.option(
    "--base-metadata", type=click.Path(exists=True),
    default=os.path.join(
        SCALING_DEFAULTS,
        "metadata_template.j2"
    ),
    help="Base metadata jinja2 template", show_default=True
)
@click.option(
    "--inventory", type=click.Path(exists=True),
    default=os.path.join(
        SCALING_DEFAULTS,
        "inventory_template.j2"
    ),
    help="Base inventory jinja2 template", show_default=True
)
@click.option(
    "--ansible-install", type=click.Path(exists=True),
    default=os.path.join(
        SCALING_DEFAULTS,
        "install_primary_replicas.j2"
    ),
    help="Ansible-freeipa install jinja2 template", show_default=True
)
@click.option(
    "--controller", type=click.Path(exists=True),
    default=os.path.join(
        SCALING_DEFAULTS,
        "controller.j2"
    ),
    help="Controller metadata file jinja2 template", show_default=True
)
@click.option(
    "--out-dir", "-o",
    type=str, default="FILES",
    help="Output directory to store generated data"
)
@click.option(
    "--metadata-storage",
    type=str, default="idm-artifacts.usersys.redhat.com",
    help="Metadata storage server"
)
@click.option(
    "--idm-ci",
    type=str,
    default="https://gitlab.cee.redhat.com/identity-management/idm-ci.git",  # FIXME
    help="IDM-CI repo"
)
@click.option(
    "--repo-branch",
    type=str,
    default="master",
    help="IDM-CI repo branch"
)
@click.option(
    "--tool-repo",
    type=str,
    default="https://gitlab.cee.redhat.com/identity-management/idm-performance-testing.git",  # FIXME
    help="SCALE tool repo"
)
@click.option(
    "--tool-branch",
    type=str,
    default="master",
    help="SCALE tool repo branch"
)
@click.option(
    "--node-os",
    type=str,
    default="fedora-33",
    help="Host operating system"
)
@click.option(
    "--project", "-p",
    type=str, default="trigger_performance_scale/large-scale",
    help="Pipeline project for storing data"
)
@click.option(
    "--run", "-r",
    type=str, default="RUNNING",
    help="Pipeline run for storing data"
)
@click.option(
    "--job", "-j",
    type=str, default="JOB",
    help="Pipeline job for storing data"
)
@click.option(
    "--freeipa-upstream-copr",
    type=str,
    help="freeipa-upstream-copr"
)
@click.option(
    "--freeipa-downstream-copr",
    type=str,
    help="freeipa-downstream-copr"
)
@click.option(
    "--freeipa-custom-repo",
    type=str,
    help="freeipa-custom-repo"
)
@click.option(
    "--ansible-freeipa-upstream-copr",
    type=str,
    help="ansible-freeipa-upstream-copr"
)
@click.option(
    "--ansible-freeipa-downstream-copr",
    type=str,
    help="ansible-freeipa-downstream-copr"
)
@click.option(
    "--ansible-freeipa-custom-repo",
    type=str,
    help="ansible-freeipa-custom-repo"
)
@click.pass_context
def jenkins_topology(
    ctx, jenkins_template, out_dir, storage, node_os,
    idm_ci, repo_branch, tool_repo, tool_branch, project, run, job,
    metadata_storage, base_metadata, inventory, ansible_install, controller,
    freeipa_upstream_copr, freeipa_downstream_copr, freeipa_custom_repo,
    ansible_freeipa_upstream_copr, ansible_freeipa_downstream_copr,
    ansible_freeipa_custom_repo,
):
    ctx = load_context(ctx, storage)
    print()
    try:
        G = ctx.obj[GRAPH_NX]
        master = ctx.obj[MASTER]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    all_successors = list(nx.bfs_successors(G, master))
    levels = levels_from_fist_successors(all_successors[:1])
    successors = all_successors[1:]

    new = []
    predecessors = predecessors_from_first_levels(levels)

    for s, f in successors:
        # print("------------------------------------")
        # print("s:", s, "f:", f, "level:", levels[-1])
        if s in levels[-1]:
            # print(s, f)
            new = new + f
            # print("+")
        else:
            # print(s, f)
            # print("next", f, "level:", new)
            # print("before", levels)
            levels.append(new)
            # print("afeter", levels)
            new = f
        for pred in f:
            predecessors[pred] = s

    if new:  # add level at the end of for cycle so last is added
        levels.append(new)

    key = 0
    levels_dict = {}
    for level in levels:
        levels_dict[key] = level
        # print(level)
        key += 1

    backbone_edges = compatible_backbone_edges(G, master)
    ctx.obj[LEVELS] = levels_dict
    ctx.obj[PRED] = predecessors
    ctx.obj[BACKBONE] = backbone_edges
    ctx.obj[EDGES] = set(G.edges)

    levels = levels_dict

    print(f"Create temporary {out_dir} folder")
    os.makedirs(out_dir, exist_ok=True)

    print("Generate Jenkinsfile for whole topology")
    output_jenkins_job = os.path.join(out_dir, "Jenkinsfile")

    pipeline_template = load_jinja_template(jenkins_template)

    jenkinsfile = pipeline_template.render(
        levels=levels,
        node_os=node_os,
        idm_ci=idm_ci,
        repo_branch=repo_branch,
        metadata_storage=metadata_storage,
        project=project,
        run=run,
        job=job,
        freeipa_upstream_copr=freeipa_upstream_copr,
        freeipa_downstream_copr=freeipa_downstream_copr,
        freeipa_custom_repo=freeipa_custom_repo,
        ansible_freeipa_upstream_copr=ansible_freeipa_upstream_copr,
        ansible_freeipa_downstream_copr=ansible_freeipa_downstream_copr,
        ansible_freeipa_custom_repo=ansible_freeipa_custom_repo
    )
    save_data(output_jenkins_job, jenkinsfile)

    print_topology(levels)
    # merge dictionary items to one list containing all nodes
    topo_nodes = list(chain(*[levels[level] for level in levels]))

    print("Generate metadata for each topology node")
    # Load jinja teplate
    metadata_template = load_jinja_template(base_metadata)

    # Generate the metadata files for each node inside of FILES directory
    # gen_metadata(template, "y0") etc.
    for node in topo_nodes:
        gen_metadata(node, node_os, out_dir,
                     project, run, job,
                     metadata_template,
                     tool_repo, tool_branch)

    print("Generate ansible-freeipa inventory file")
    # Load jinja teplate
    job_inventory = load_jinja_template(inventory)

    # Generate ansible-freeipa inventory file
    outpujob_inventory = os.path.join(out_dir, "perf-inventory")
    inventoryfile = job_inventory.render(master_server=topo_nodes[0],
                                         levels=levels,
                                         predecessors=predecessors)
    save_data(outpujob_inventory, inventoryfile)

    print("Generate ansible-freeipa install file")
    missing_edges = set(G.edges) - set(backbone_edges)

    segments = get_missing_segments(missing_edges)
    # Load jinja teplate
    ansible_install = load_jinja_template(ansible_install)
    # Generate ansible-freeipa install file
    output_install = os.path.join(out_dir, "perf-install.yml")
    installfile = ansible_install.render(levels=levels,
                                         missing=segments)
    save_data(output_install, installfile)

    print("Generate controler metadata")
    controller_templ = load_jinja_template(controller)
    # Generate controler metadata
    controler_file = controller_templ.render(
        metadata_storage=metadata_storage,
        node_os=node_os,
        project=project,
        run=run,
        job=job,
        tool_repo=tool_repo,
        tool_branch=tool_branch,
    )
    output_controller = os.path.join(out_dir, "controller.yaml")
    save_data(output_controller, controler_file)


def compatible_backbone_edges(G, master):
    """Rerurn edges not based on direction of bfs walk"""
    all_edges = set(G.edges)
    b_edges = set(nx.bfs_edges(G, master))

    backbone_edges = set()
    for e in b_edges:
        if e in all_edges:
            backbone_edges.add(e)
        else:
            backbone_edges.add((e[1], e[0]))

    return backbone_edges


def add_list_item(record, item):
    if record is None:
        record = []

    record.append(item)
    return record


def sort_by_degree(G, nodes, reverse=False):
    sorted_nodes = OrderedDict()
    nodes_by_degree = {}
    for node in nodes:
        cnt = nx.degree(G, node)
        try:
            record = nodes_by_degree[cnt]
        except KeyError:
            record = []
        nodes_by_degree[cnt] = add_list_item(record, node)

    for degree in range(max(nodes_by_degree, key=int), 0, -1):
        try:
            sorted_nodes[degree] = sorted(
                nodes_by_degree[degree], reverse=reverse
            )
        except KeyError:
            pass

    return sorted_nodes


@graphcli.command()
@click.option("-o", "--output", default="./topology_playbook")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def analyze(ctx, storage, output):
    ctx = load_context(ctx, storage)

    try:
        G = ctx.obj[GRAPH_NX]
        # master = ctx.obj[MASTER]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    print("---------------------------------------")
    print("General graph information:\n" + str(nx.info(G)))
    print("Connected components:\t" + str(nx.number_connected_components(G)))
    print("---------------------------------------")
    print("Analyzing nodes as standalone entity...")
    print("---------------------------------------")
    print("ISSUE\t\t\t\t\t| NODE")
    print("-----\t\t\t\t\t| ----")
    for node in G:
        repl_agreements = nx.degree(G, node)

        if repl_agreements < 2:
            print(f"Less than 2 replication agreemenst\t| {node}")

        if repl_agreements > 4:
            print(f"More than 4 replication agreemenst\t| {node}")

    print("---------------------------------------")
    print("Analyzing topology connectivity...")
    print("---------------------------------------")
    print("Looking for articulation points...")

    art_points = sorted(list(nx.articulation_points(G)), reverse=True)
    for art_point in art_points:
        print(f"Articulation point\t| {art_point}")

    components = list(nx.biconnected_components(G))

    if art_points:
        print(f"Articulation point(s) found: {len(art_points)}")
        print("---------------------------------------")
        print("Looking for biconected components...")

        for comp in components:
            print(f"Biconected component: {comp}")

        print(f"Biconected component(s) found: {len(components)}")

    print("---------------------------------------")


@graphcli.command()
@click.option("-o", "--output", default="./topology_playbook")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def fixup(ctx, storage, output):
    ctx = load_context(ctx, storage)

    try:
        G = ctx.obj[GRAPH_NX]
        # master = ctx.obj[MASTER]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    candidates = []

    art_points = sorted(list(nx.articulation_points(G)), reverse=True)
    components = list(nx.biconnected_components(G))

    print("================================================")

    # trying to remove articulation points
    added_edges = []

    while True:
        if len(list(nx.articulation_points(G))) == 0:
            # if there are no articulation points we stop
            break

        if not art_points:
            # no points left to take care of from previous cycle
            art_points = list(nx.articulation_points(G))

        art_point = art_points.pop()

        for comp in components:
            # print("art points:", art_points)
            # print("processing:", art_point)
            if art_point not in comp:
                # skip if articulation point is not in component
                continue

            # we would pick node with the lowest degree to add edge to

            sorted_nodes = sort_by_degree(G, comp, reverse=True)

            min_degree = min(sorted_nodes, key=int)

            if min_degree < 4:
                print(sorted(sorted_nodes[min_degree]))
                candidates.append(sorted(sorted_nodes[min_degree])[0])
            else:
                nx.draw(G, with_labels=True, font_weight='bold')
                plt.show()
                raise Error(
                    f"Error: Can not add edge to component ({comp}) with nodes "
                    f"that have 4 or more replication agreements ({min_degree})."
                )

            # if you can just connect and remember right node as new left

            edges_to_add = []

            if len(candidates) >= 2:
                left = candidates[-2]
                right = candidates[-1]
                edges_to_add.append((left, right))
                G.add_edges_from(edges_to_add, color="green")
                added_edges += edges_to_add

    print("---------------------------------------")
    print("Added edges:")
    for edge in added_edges:
        print(edge)

    print(f"Added edge(s) count: {len(added_edges)}")
    print("---------------------------------------")

    colors = []
    for u, v in G.edges():
        try:
            colors.append(G[u][v]["color"])
        except KeyError:
            colors.append("black")

    nx.draw(G, with_labels=True, font_weight='bold', edge_color=colors)
    plt.show()

    # we ecpect that previous cycle got newG to be same as G
    newG = nx.Graph()
    newG.add_edges_from(G.edges)
    assert nx.is_isomorphic(G, newG)

    while nx.is_isomorphic(G, newG):
        break

    # save new graph
    ctx.obj[GRAPH_NX] = G
    with shelve.open(storage) as db:
        for key in ctx.obj:
            db[key] = ctx.obj[key]



if __name__ == "__main__":
    sys.exit(graphcli())
