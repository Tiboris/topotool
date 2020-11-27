#!/usr/bin/env python3
import os
import click
import sys
import shelve
import networkx as nx
import matplotlib.pyplot as plt


from graph_tool.graphs import Graph
from copy import deepcopy
from matplotlib.pyplot import colorbar
from jinja2 import Template


GRAPH_DATA = "graph_input_data"
GRAPH_OBJECT = "graph_object"
GRAPH_NX = "graph_networkx"
LEVELS = "levels"
PRED = "predecessors"
BACKBONE = "backbone_edges"
EDGES = "edges"
MASTER = "master_node"


def load_context(ctx, storage):
    ctx.ensure_object(dict)
    if not ctx.obj:
        try:
            with shelve.open(storage) as db:
                ctx.obj[GRAPH_DATA] = db[GRAPH_DATA]
                ctx.obj[GRAPH_OBJECT] = db[GRAPH_OBJECT]
                ctx.obj[GRAPH_NX] = db[GRAPH_NX]
                ctx.obj[LEVELS] = db[LEVELS]
                ctx.obj[PRED] = db[PRED]
                ctx.obj[MASTER] = db[MASTER]
                ctx.obj[EDGES] = db[EDGES]
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

    topo["predecessors"] = predecessors
    topo["backbone_edges"] = [
        (value, key) for key, value in predecessors.items()
    ]

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
@click.option("-d", "--delimiter", default="-")
@click.option("-w", "--weight-delimiter", default="")
@click.option("-m", "--master", default="y0")
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def load(ctx, input_file, delimiter, weight_delimiter, master, storage):
    """Loads grap data to dictionary"""
    ctx.ensure_object(dict)

    with open(input_file, "r") as f:
        ctx.obj[GRAPH_DATA] = f.read().splitlines()

    data = deepcopy(ctx.obj[GRAPH_DATA])
    graph_data = Graph(
        data, delimiter=delimiter, weight_delim=weight_delimiter
    )
    ctx.obj[GRAPH_OBJECT] = graph_data

    G = nx.Graph()
    G.name = "Complete graph of topology"
    for vertex in graph_data.vertices:
        G.add_node(vertex)

    G.add_edges_from(graph_data.edge_list)
    ctx.obj[GRAPH_NX] = G

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
    level_dict = {}
    for level in levels:
        level_dict[key] = level
        # print(level)
        key += 1

    # for level, replicas in level_dict.items():  # for inventory
    #     print("---")
    #     print(level, replicas)
    #     if level:
    #         for replica in replicas:
    #             print(replica, predecessors[replica])

    ctx.obj[LEVELS] = level_dict
    ctx.obj[PRED] = predecessors
    ctx.obj[MASTER] = master
    ctx.obj[BACKBONE] = compatible_backbone_edges(G, master)
    ctx.obj[EDGES] = set(G.edges)
    print("Loading of topology done.")

    with shelve.open(storage) as db:
        for key in ctx.obj:
            # print(key)
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
@click.pass_context
def generate(ctx, storage, branches, length, nodes):
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

    ctx.obj[LEVELS] = topo["levels"]
    print(topo["levels"])

    ctx.obj[PRED] = topo["predecessors"]
    ctx.obj[MASTER] = "y0"
    ctx.obj[BACKBONE] = topo["backbone_edges"]
    for edge in topo["edges"]:
        print(edge)
    ctx.obj[EDGES] = topo["edges"]
    print("Generating of topology done.")

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


@graphcli.command()
@click.option("-o", "--output", default="./topology_playbook")
@click.option("-d", "--out-dir", default="./FILES")
@click.option("-t", "--template", default="../data/inventory_template.j2")  # FIXME
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def inventory(ctx, storage, out_dir, template, output):
    ctx = load_context(ctx, storage)

    try:
        pred = ctx.obj[PRED]
        levels = ctx.obj[LEVELS]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    print(f"Create temporary {out_dir} folder")
    os.makedirs(out_dir, exist_ok=True)

    print(levels, pred)


@graphcli.command()
@click.option("-o", "--output", default="./graph_inventory")
@click.option("-d", "--out-dir", default="./FILES")
@click.option("-t", "--template", default="../data/jenkinsjob.j2")  # FIXME
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def jenkinsjob(ctx, output, template, out_dir, storage):
    ctx = load_context(ctx, storage)

    try:
        levels = ctx.obj[LEVELS]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    print(f"Create temporary {out_dir} folder")
    os.makedirs(out_dir, exist_ok=True)

    print("Generate Jenkinsfile for whole topology")
    # output_jenkins_job = os.path.join(out_dir, output)

    # jenkinsfile = jenkins_template.render(
    #     levels=levels,
    #     node_os=node_os,
    #     idm_ci=idm_ci,
    #     repo_branch=repo_branch,
    #     metadata_storage=metadata_storage,
    #     project=project,
    #     run=run,
    #     job=job,
    #     freeipa_upstream_copr=freeipa_upstream_copr,
    #     freeipa_downstream_copr=freeipa_downstream_copr,
    #     freeipa_custom_repo=freeipa_custom_repo,
    #     ansible_freeipa_upstream_copr=ansible_freeipa_upstream_copr,
    #     ansible_freeipa_downstream_copr=ansible_freeipa_downstream_copr,
    #     ansible_freeipa_custom_repo=ansible_freeipa_custom_repo
    # )
    # save_data(output_jenkins_job, jenkinsfile)


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


@graphcli.command()
@click.option("-o", "--output", default="./topology_playbook")
@click.option("-d", "--out-dir", default="./FILES")
@click.option("-t", "--template", default="../data/inventory_template.j2")  # FIXME
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def playbooks(ctx, storage, out_dir, template, output):
    ctx = load_context(ctx, storage)

    try:
        pred = ctx.obj[PRED]
        levels = ctx.obj[LEVELS]
        backbone_edges = ctx.obj[BACKBONE]
        all_edges = ctx.obj[EDGES]
    except KeyError:
        print("Please load or generate the topology first.")
        exit(1)

    print(f"Create temporary {out_dir} folder")
    os.makedirs(out_dir, exist_ok=True)

    missing_edges = all_edges.difference(backbone_edges)

    # print(all_edges, len(all_edges))
    # print(backbone_edges, len(backbone_edges))
    # print(missing_edges, len(missing_edges))

    missing = nx.Graph()
    # print(dir(missing))
    missing.name = "Graph with red missing edges from backbone"
    missing.add_edges_from(backbone_edges, color="black")
    missing.add_edges_from(missing_edges, color="red")
    colors = [missing[u][v]['color'] for u, v in missing.edges()]

    with shelve.open(storage) as db:
        for key in ctx.obj:
            db[key] = ctx.obj[key]

    nx.draw(missing, with_labels=True, font_weight='bold', edge_color=colors)
    # nx.draw_planar(
    #     missing, with_labels=True, font_weight='bold', edge_color=colors
    # )

    res = {
        LEVELS: levels,
        PRED: pred,
        BACKBONE: backbone_edges,
        EDGES: all_edges,
    }

    # print(res)
    plt.show()

    return res


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

    print("Articulation points:", list(nx.articulation_points(G)))

    print("Biconected components:", list(nx.biconnected_components(G)))
    print("Biconected component edges:", list(nx.biconnected_component_edges(G)))
    print("Connected components:", str(nx.number_connected_components(G)))

    for node in G:
        print("Degree of Node in graph:", f"{node}:", str(nx.degree(G, node)))

    print("===\nGraph info:\n" + str(nx.info(G)))


if __name__ == "__main__":
    sys.exit(graphcli())
