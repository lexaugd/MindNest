## Authentication

Mechanisms are not included in this specification to manage
authentication to external services. However, the working group has shown 
multiple solutions to this as part of cloudmesh. This includes the posibility
of a 

* *Local configuration file:* A configuration file is managed locally to
  allow access to the clouds. It is the designer's responsibility
  not to expose such credentials.
* *Session based authentication:* No passwords are stored in the
  configuration file and access is granted on a per session basis where
  the password needs to be entered.
* *Service based authentication:* The authentication is delegated to an
  external process. The service that acts on behalf
  of the user needs to have access to the appropriate cloud provider 
  credentials

An example for a configuration file is provided at [@cloudmesh4-yaml].


