#!/usr/bin/env python3
import os

import click
import sys
import shelve

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from numpy import sqrt
from collections import Counter, OrderedDict
from topotool.graphs import Graph
from copy import deepcopy
from jinja2 import Template
from itertools import chain

MAX_REPL_AGREEMENTS = 4
MIN_REPL_AGREEMENTS = 2

GRAPH_DATA = "graph_input_data"
GRAPH_OBJECT = "graph_object"
GRAPH_NX = "graph_networkx"
MASTER = "master_node"

DB_KEYS = [
    GRAPH_DATA,
    GRAPH_OBJECT,
    GRAPH_NX,
    MASTER,
]

SCALING_DEFAULTS = os.path.join(
    os.path.dirname(__file__),
    "./data/"
)


def load_context(ctx, storage):
    ctx.ensure_object(dict)
    if not ctx.obj:
        for db_key in DB_KEYS:
            try:
                with shelve.open(storage) as db:
                    ctx.obj[db_key] = db[db_key]
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


def gen_metadata(topo_nodes, node_os, destination,
                 project, run, job, template,
                 tool_repo, tool_branch):
    metadata = template.render(
        topo_nodes=topo_nodes,
        node_os=node_os,
        project=project,
        run=run,
        job=job,
        tool_repo=tool_repo,
        tool_branch=tool_branch
    )
    output_file = os.path.join(destination, "scaling_metadata.yaml")
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

    ctx.obj[GRAPH_OBJECT] = graph_data

    G = nx.Graph()
    G.name = f"Loaded graph of topology from {input_file}"
    for vertex in graph_data.vertices:
        G.add_node(vertex)

    G.add_edges_from(graph_data.edge_list)

    ctx.obj[GRAPH_NX] = G
    ctx.obj[MASTER] = master

    sys.stdout.write("Loading of topology done\n")

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
    ctx.obj[MASTER] = master

    with shelve.open(storage) as db:
        for key in ctx.obj:
            db[key] = ctx.obj[key]


def produce_output_image(G, filename):
    plt.close()

    art_points = list(nx.articulation_points(G))
    # https://stackoverflow.com/questions/49121491/issue-with-spacing-nodes-in-networkx-graph
    pos = nx.spring_layout(
        G, k=0.3*1/sqrt(len(G.nodes())), iterations=150
    )

    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.3
    plt.xlim(x_min - x_margin, x_max + x_margin)

    plot_nodes = {
        "overloaded_art_points": {
            "nodes": [],
            "color": "orange",
        },
        "art_points": {
            "nodes": [],
            "color": "yellow",
        },
        "overloaded": {
            "nodes": [],
            "color": "red",
        },
        "one_edge": {
            "nodes": [],
            "color": "pink",
        },
        "complying_nodes": {
            "nodes": [],
            "color": "green",
        },
    }

    for node in G:
        node_degree = nx.degree(G, node)
        if node in art_points and node_degree > 4:
            plot_nodes["overloaded_art_points"]["nodes"].append(node)
        elif node in art_points:
            plot_nodes["art_points"]["nodes"].append(node)
        elif node_degree > 4:
            plot_nodes["overloaded"]["nodes"].append(node)
        elif node_degree == 1:
            plot_nodes["one_edge"]["nodes"].append(node)
        else:
            plot_nodes["complying_nodes"]["nodes"].append(node)

    for node_type in plot_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=plot_nodes[node_type]["nodes"],
            node_color=plot_nodes[node_type]["color"],
            node_size=500,
            alpha=0.8
        )

    # set colors for the edges to draw
    colors = []
    for u, v in G.edges():
        try:
            colors.append(G[u][v]["color"])
        except KeyError:
            colors.append("black")

    nx.draw_networkx_edges(
        G, pos,
        edge_color=colors,
        edgelist=G.edges
    )

    labels = {}

    for node in G:
        labels[node] = node

    nx.draw_networkx_labels(G, pos, labels)

    patches = []
    for node_type in plot_nodes:
        patches.append(
            mpatches.Patch(
                color=plot_nodes[node_type]["color"], label=node_type
            )
        )

    plt.legend(handles=patches)

    plt.axis('off')
    if filename:
        plt.savefig(
            os.path.abspath(filename),
            dpi=400
        )
    else:
        plt.show()


@graphcli.command()
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def draw(ctx, storage):
    """Draw a topology graph as picture."""
    ctx = load_context(ctx, storage)

    try:
        G = ctx.obj[GRAPH_NX]
    except KeyError:
        sys.stderr.write("Please load or generate the topology first\n")
        sys.exit(1)

    produce_output_image(G, filename="topology_drawing.png")


def get_segments(edges):
    segments = {}
    nodes = list(chain(*edges))
    most_common_nodes = Counter(nodes).most_common()
    node_iter = iter(most_common_nodes)
    processed = set()
    not_processed = set(edges)
    while not_processed:
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
)
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
    metadata_storage, base_metadata, inventory, ansible_install,
    freeipa_upstream_copr, freeipa_downstream_copr, freeipa_custom_repo,
    ansible_freeipa_upstream_copr, ansible_freeipa_downstream_copr,
    ansible_freeipa_custom_repo,
):
    ctx = load_context(ctx, storage)

    try:
        G = ctx.obj[GRAPH_NX]
        master = ctx.obj[MASTER]
    except KeyError:
        sys.stderr.write("Please load or generate the topology first\n")
        sys.exit(1)

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
        key += 1

    backbone_edges = compatible_backbone_edges(G, master)
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

    print("Generate metadata for topology nodes")
    # Load jinja teplate
    metadata_template = load_jinja_template(base_metadata)

    # Generate the metadata file for all nodes inside of FILES directory
    gen_metadata(topo_nodes, node_os, out_dir,
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

    segments = get_segments(missing_edges)
    # Load jinja teplate
    ansible_install = load_jinja_template(ansible_install)
    # Generate ansible-freeipa install file
    output_install = os.path.join(out_dir, "perf-install.yml")
    installfile = ansible_install.render(levels=levels,
                                         missing=segments)
    save_data(output_install, installfile)

    produce_output_image(
        G, os.path.join(out_dir, "topology.png")
    )


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


def sort_by_degree(G, nodes=None, reverse=False):
    sorted_nodes = OrderedDict()
    nodes_by_degree = {}

    if not nodes:
        nodes = G

    for node in nodes:
        cnt = nx.degree(G, node)
        try:
            record = nodes_by_degree[cnt]
        except KeyError:
            record = []
        nodes_by_degree[cnt] = add_list_item(record, node)

    if reverse:
        for degree in range(max(nodes_by_degree, key=int), 0, -1):
            try:
                sorted_nodes[degree] = sorted(
                    nodes_by_degree[degree], reverse=reverse
                )
            except KeyError:
                pass
    else:
        for degree in range(max(nodes_by_degree, key=int) + 1):
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
        sys.stderr.write("Please load or generate the topology first.")
        sys.exit(1)

    print("----------------------------------------")
    print("General graph information:\n" + str(nx.info(G)))
    print("Connected components:\t" + str(nx.number_connected_components(G)))
    print("----------------------------------------")
    print("Analyzing nodes as standalone entity...")
    print("--------------------")
    issues = []
    for node in G:
        repl_agreements = nx.degree(G, node)

        if repl_agreements < 2:
            issues.append(f"Less than 2 replication agreemenst\t| {node}")

        if repl_agreements > MAX_REPL_AGREEMENTS:
            issues.append(
                f"More than {MAX_REPL_AGREEMENTS} "
                f"replication agreemenst\t| {node}"
            )

    if issues:
        print("ISSUE\t\t\t\t\t| NODE")
        print("-----\t\t\t\t\t| ----")
        for issue in issues:
            print(issue)
    else:
        print("No issues found for the nodes.")

    print("----------------------------------------")
    print("Analyzing topology connectivity...")
    print("--------------------")
    print("Looking for articulation points...")

    art_points = sorted(list(nx.articulation_points(G)), reverse=True)
    components = list(nx.biconnected_components(G))

    if art_points or components:
        for art_point in art_points:
            print(f"Articulation point\t| {art_point}")

        print(f"Articulation point(s) found: {len(art_points)}")
        print("--------------------")
        print("Looking for biconected components...")

        for comp in components:
            print(f"Biconected component: {comp}")

        print(f"Biconected component(s) found: {len(components)}")

    print("----------------------------------------")


def remove_articulation_points(G, step, omit_max_degree, max_repl_agreements):

    # trying to remove articulation points
    added_edges = []
    can_not_add = []
    to_add = ""

    all_components = list(nx.biconnected_components(G))

    while list(nx.articulation_points(G)):
        art_points = sorted(list(nx.articulation_points(G)), reverse=True)

        if len(list(nx.articulation_points(G))) == 0:
            # if there are no articulation points we stop
            sys.stdout.write("Info: Will not add more edges\n")
            break

        # pick articulation point to solve issue for FIXME
        art_point = art_points.pop()

        color = "green"  # default added edge color

        # we build list of candidates
        candidates = []

        while len(candidates) != 2:
            if not all_components:
                break

            try:
                comp = all_components.pop()
            except IndexError:
                sys.stderr.write("No more components left to try connecting\n")
                sys.exit(2)

            # get dictionary with keys corresponding to degrees and each key
            # will have a list of node with following degree as value
            sorted_nodes = sort_by_degree(G, nodes=comp, reverse=False)

            # we would choose a node with the fewest connection count
            # therefore the lowerest degree of the node

            to_pick_from = []
            # list with nodes, first in the list are with the lower degree
            for _degree, nodes in sorted_nodes.items():
                to_pick_from += nodes

            start = 0
            to_add = None
            while (
                not to_add  # and to_add not in art_points
                and (start < len(to_pick_from))
            ):
                to_add = to_pick_from[start]
                # print(to_add, to_pick_from)
                start += 1

            min_degree = nx.degree(G, to_add)

            if min_degree < max_repl_agreements:
                candidates.append(to_add)
                color = "green" if len(candidates) > 2 else color
            else:
                added = False
                if omit_max_degree:
                    candidates.append(to_add)
                    color = "orange"
                    added = True
                else:
                    can_not_add.append(to_add)

                added_str = "Added" if added else "Did not add"
                sys.stdout.write(
                    f"Warning: {added_str} {color} edge to connect component "
                    f"({comp}) with articulation point {art_point} "
                    f"{to_add} has {max_repl_agreements} "
                    "or more replication agreements\n"
                )

                if not added and to_add in can_not_add:
                    sys.stdout.write(
                        "Hint: Try adding --omit-max-degree option to add "
                        "edges even when maximum degree of node is reached\n"
                    )

            if to_add in can_not_add:
                neighs = G.neighbors(to_add)
                sys.stderr.write(
                    f"Unable to add more replication agreements to: {to_add}\n"
                )
                for nei in neighs:
                    sys.stderr.write(f"{to_add} has neihgbor: {nei}\n")
                sys.exit(3)

        if len(candidates) == 2:
            left = candidates[-2]
            right = candidates[-1]
            edge_to_add = (left, right)
            step += 1
            G.add_edges_from([edge_to_add], color=color)
            produce_output_image(
                G, f"fixup_step_{step}_add_{left}_{right}.png"
            )
            added_edges.append(edge_to_add)

        all_components = list(nx.biconnected_components(G))

    print("----------------------------------------")
    print("Added edges:")
    for edge in added_edges:
        print(edge)

    print(f"Added edge(s) count: {len(added_edges)}")
    print("----------------------------------------")

    return G, added_edges, step


def remove_overloaded_nodes_edges(
    G, step, add_while_removing, omit_max_degree, max_repl_agreements
):
    removed = []
    can_not_remove = []
    remove = set()
    added_edges = []  # used when add_while_removing is set to true

    sorted_nodes = sort_by_degree(G)
    max_degree = max(sorted_nodes, key=int)

    while max_degree > max_repl_agreements:
        # sort the G nodes by degree once!
        sorted_max_degree_nodes = iter(sorted_nodes[max_degree])

        # for max_degree_node in sorted_max_degree_nodes:
        max_degree_node = next(sorted_max_degree_nodes)

        if max_degree <= max_repl_agreements:
            sys.stdout.write("Info: Will not remove more edges\n")
            break

        neighbors = sorted(
            [n for n in G.neighbors(max_degree_node)]
        )

        sort_neig_dict = sort_by_degree(G, nodes=neighbors)
        sort_neig = sorted(sort_neig_dict, reverse=True)

        remove = removed[-1] if removed else remove

        for neig in sort_neig_dict[sort_neig[0]]:
            to_remove = (max_degree_node, neig)

            if to_remove not in can_not_remove:
                remove = to_remove
                break

        if remove in can_not_remove:
            sys.stderr.write(
                "Error: Can not remove any edge without "
                "creating articulation points\n"
            )
            sys.stderr.write(
                "Try to use option '--add-while-removing' to remove "
                "articulation poins created by this removal\n"
            )
            sys.exit(4)

        if remove in removed:
            sys.stderr.write(f"Error: Edge already removed - {remove}\n")
            sys.exit(4)

        if remove:
            step += 1
            G.remove_edge(*remove)

        produce_output_image(
            G, filename=f"fixup_step_{step}_rm_{remove[0]}_{remove[1]}.png"
        )

        sorted_nodes = sort_by_degree(G)
        max_degree = max(sorted_nodes, key=int)

        removed.append(remove)

        if len(list(nx.articulation_points(G))) == 0:
            # if there are no articulation points we continue removing
            continue
        else:
            sys.stdout.write(
                f"Warning: Removal of the {removed[-1]} edge created "
                "articulation point\n"
            )
            if add_while_removing:
                sys.stdout.write(
                    "Info: Trying to fix new articulation points\n"
                )
                G, added, step = remove_articulation_points(
                    G, step=step,
                    omit_max_degree=omit_max_degree,
                    max_repl_agreements=max_repl_agreements,
                )
                added_edges += added

                for new in added:
                    if (
                        (new[0], new[1]) in removed
                        or (new[1], new[0]) in removed
                    ):
                        sys.stderr.write(
                            f"Error Added edge {new} has been found in"
                            f"already removed list: {removed}\n"
                        )
                        sys.exit(4)

            else:
                sys.stdout.write(f"Info: Adding edge {removed[-1]} back\n")
                G.add_edge(removed[-1][0], removed[-1][1])
                can_not_remove.append(removed[-1])
                sys.stdout.write(f"list to can not remove: {can_not_remove}")

    return G, removed, added_edges, step


@graphcli.command()
@click.option(
    "-h", "--max", "max_repl_agreements",
    default=MAX_REPL_AGREEMENTS, type=click.IntRange(2, 10),
)
@click.option("--omit-max-degree", is_flag=True, default=False)
@click.option("--add-while-removing", is_flag=True, default=False)
@click.option("-s", "--storage", default="./.graph_storage.json")
@click.pass_context
def fixup(ctx, storage, max_repl_agreements,
          omit_max_degree, add_while_removing):
    ctx = load_context(ctx, storage)

    try:
        G = ctx.obj[GRAPH_NX]
    except KeyError:
        sys.stderr.write("Please load or generate the topology first\n")
        sys.exit(1)

    print("========================================")
    step = 0
    produce_output_image(G, f"fixup_step_{step}_starting_with.png")

    G, added_edges, step = remove_articulation_points(
        G, step=step,
        omit_max_degree=omit_max_degree,
        max_repl_agreements=max_repl_agreements,
    )

    G, removed, added_with_removal, step = remove_overloaded_nodes_edges(
        G, step=step,
        add_while_removing=add_while_removing,
        omit_max_degree=omit_max_degree,
        max_repl_agreements=max_repl_agreements,
    )

    added_edges += added_with_removal

    missing_segments = get_segments(added_edges)
    redundant_segments = get_segments(removed)

    print("----------------------------------------")
    print("Removed edges:")
    for edge in removed:
        print(edge)

    print(f"Removed edge(s) count: {len(removed)}")
    print("----------------------------------------")

    print("Summary:")

    if missing_segments or redundant_segments:
        print(f"Added edge(s) [{len(added_edges)}]:")
        for edge in added_edges:
            print(edge)
        print(f"Removed edge(s) [{len(removed)}] :")
        for edge in removed:
            print(edge)
        # Difference should be empty
        print("Difference:", set(added_edges).intersection(set(removed)))

        print("----------------------------------------")

        print("Saving result graph image")

        produce_output_image(
            G, "fixup_result.png"
        )

        print("----------------------------------------")

        print("Saving result graph data")

        # save new graph to context db
        ctx.obj[GRAPH_NX] = G
        with shelve.open(storage) as db:
            for key in ctx.obj:
                db[key] = ctx.obj[key]

        print("----------------------------------------")
        print("Generating fixup playbook")

        fixup_playbook = load_jinja_template(os.path.join(
            SCALING_DEFAULTS,
            "fixup_topology_segments.j2"
        ))

        # Generate fixup playbook
        fixup = "fixup_result.yml"
        fixup_data = fixup_playbook.render(
            missing=missing_segments,
            redundant=redundant_segments,
        )

        save_data(fixup, fixup_data)
    else:
        print("Nothing to do...")

    print("========================================")


if __name__ == "__main__":
    sys.exit(graphcli())
