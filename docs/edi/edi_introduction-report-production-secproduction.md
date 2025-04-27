## Report Production {#sec:production}


The *NBDIF: Volume 8,* *References Architecture Interfaces* is one of
nine volumes, whose overall aims are to define and prioritize Big Data
requirements, including interoperability, portability, reusability,
extensibility, data usage, analytic techniques, and technology
infrastructure to support secure and effective adoption of Big Data. The
overall goals of this volume are to define and specify interfaces to
implement the Big Data Reference Architecture. This volume arose from
discussions during the weekly NBD-PWG conference calls. Topics included
in this volume began to take form in Phase 2 of the NBD-PWG work. During 
the discussions, the NBD-PWG identified the need to
specify a variety of interfaces.


Phase 3 work, which built upon the groundwork developed during Phase 2, 
included an early specification based on resource object specifications 
that provided a simplified version of an API interface design.

Report Structure
----------------

To enable interoperability between the NBDRA components, a list of
well-defined NBDRA interfaces is needed. These interfaces are documented
in this volume. To introduce them, the NBDRA structure will be followed,
focusing on interfaces that allow bootstrapping of the NBDRA. The
document begins with a summary of requirements that will be integrated
into our specifications. Subsequently, each section will introduce a
number of objects that build the core of the interface addressing a
specific aspect of the NBDRA. A selected number of *interface use cases*
will be showcased to outline how the specific interface can be used in a
reference implementation of the NBDRA. Validation of this approach can
be achieved while applying it to the application use cases that have
been gathered in the *NBDIF: Volume 3, Use Cases and Requirements*
document. These application use cases have considerably contributed
towards the design of the NBDRA. Hence the expectation is that: (1) the
interfaces can be used to help implement a Big Data architecture for a
specific use case; and (2) the proper implementation. This approach can
facilitate subsequent analysis and comparison of the use cases.

The organization of this document roughly corresponds to the process
used by the NBD-PWG to develop the interfaces. Following the
introductory material presented in @sec:introduction, the remainder of this
document is organized as follows:

* @sec:interface-requirements presents the interface requirements;
* @sec:spec-paradigm presents the specification paradigm that is used;
* @sec:specification presents several objects grouped by functional
  use while providing a summary table of selected proposed objects in
  @sec:spec-table.

While each NBDIF volume was created with a specific focus within Big
Data, all volumes are interconnected. During creation, the volumes
gave and/or received input from other volumes. Broad topics (e.g.,
definition, architecture) may be discussed in several volumes with the
discussion circumscribed by the volume’s particular focus. Arrows
shown in @fig:nist-doc-nav indicate the main flow of
input/output. Volumes 2, 3, and 5 (blue circles) are essentially
standalone documents that provide output to other volumes (e.g., to
Volume 6). These volumes contain the initial situational awareness
research. Volumes 4, 7, 8, and 9 (green circles) primarily received
input from other volumes during the creation of the particular
volume. Volumes 1 and 6 (red circles) were developed using the initial
situational awareness research and continued to be modified based on
work in other volumes. These volumes also provided input to the green
circle volumes.

![NBDIF Documents Navigation Diagram Provides Content Flow Between Volumes](images/nist-doc-nav.png){#fig:nist-doc-nav}


