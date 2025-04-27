## Identity {#sec:spec-identity}

As part of services an identity often needs to be specified. In
addition, such persons [@www-eduperson] are often part of groups. Thus, 
three important terms related to the identity are distinguished as follows:

* Organization: The information representing an Organization that
  manages a Big Data Service (@sec:spec-organization)
* Group: A group that a person may belong to that is important to
  define access to services (included in @sec:spec-organization)
* User: The information identifying the profile of a person (@sec:spec-user)

### Organization {#sec:spec-organization}

{include=./spec/organization.md}

#### organization.yaml

```{include=./spec/organization.yaml}
```

### User {#sec:spec-user}

{include=./spec/user.md}

#### user.yaml

```{include=./spec/user.yaml}
```

### Account {#sec:spec-account}

{include=./spec/account.md}

#### account.yaml

```{include=./spec/account.yaml}


### Public Key Store {#sec:spec-publickeystore}

{include=./spec/publickeystore.md}

#### publickeystore.yaml

```{include=./spec/publickeystore.yaml}
```


