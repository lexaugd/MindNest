# NBDRA Interface Requirements {#sec:interface-requirements}


The development of a Big Data reference architecture requires a thorough
understanding of current techniques, issues, and concerns. To this end,
the NBD-PWG collected use cases to gain an understanding of current
applications of Big Data, conducted a survey of reference architectures
to understand commonalities within Big Data architectures in use,
developed a taxonomy to understand and organize the information
collected, and reviewed existing technologies and trends relevant to Big
Data. The results of these NBD-PWG activities were used in the
development of the NBDRA (@fig:arch) and the interfaces presented herein.
Detailed descriptions of these activities can be found in the other
volumes of the *NBDIF*.

![NIST Big Data Reference Architecture (NBDRA)](images/bdra.png){#fig:arch}


This vendor-neutral, technology- and infrastructure-agnostic conceptual
model, the NBDRA, is shown in @fig:arch and represents a Big Data system
composed of five logical functional components connected by
interoperability interfaces (i.e., services). Two fabrics envelop the
components, representing the interwoven nature of management and
security and privacy with all five of the components. These two fabrics
provide services and functionality to the five main roles in the areas
specific to Big Data and are crucial to any Big Data solution. Note:
None of the terminology or diagrams in these documents is intended to be
normative or to imply any business or deployment model. The terms
*provider* and *consumer* as used are descriptive of general roles and
are meant to be informative in nature.

The NBDRA is organized around five major roles and multiple sub-roles
aligned along two axes representing the two Big Data value chains: the
Information Value (horizontal axis) and the Information Technology (IT;
vertical axis). Along the Information Value axis, the value is created
by data collection, integration, analysis, and applying the results
following the value chain. Along the IT axis, the value is created by
providing networking, infrastructure, platforms, application tools, and
other IT services for hosting of and operating the Big Data in support
of required data applications. At the intersection of both axes is the
Big Data Application Provider role, indicating that data analytics and
its implementation provide the value to Big Data stakeholders in both
value chains. The term *provider* as part of the Big Data Application
Provider and Big Data Framework Provider is there to indicate that those
roles provide or implement specific activities and functions within the
system. It does not designate a service model or business entity.

The DATA arrows in @fig:arch show the flow of data between the system's
main roles. Data flows between the roles either physically (i.e., by
value) or by providing its location and the means to access it (i.e., by
reference). The SW arrows show transfer of software tools for processing
of Big Data *in situ*. The Service Use arrows represent software
programmable interfaces. While the main focus of the NBDRA is to
represent the run-time environment, all three types of communications or
transactions can happen in the configuration phase as well. Manual
agreements (e.g., service-level agreements) and human interactions that
may exist throughout the system are not shown in the NBDRA.

Detailed information on the NBDRA conceptual model is presented in the
*NBDIF: Volume 6, Reference Architecture* document.

Prior to outlining the specific interfaces, general requirements are
introduced and the interfaces are defined.

