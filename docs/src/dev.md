# Developer Notes

## CompatHelper.jl, Running Tests, and Signing Commits
CompatHelper.jl will automatically look for new versions of AcousticMetrics.jl's dependencies and, if it finds any breaking versions, open PRs with changes to AcousticMetrics.jl's `Project.toml` to incorporate the new versions.
But!
The PR won't automatically run the GitHub Action tests: https://github.com/peter-evans/create-pull-request/blob/main/docs/concepts-guidelines.md#triggering-further-workflow-runs
A workaround is to manually close and immediately re-open the PR, which will run the tests and isn't too much work.

The next problem: commits created by CompatHelper.jl/the github-actions bot aren't signed, and AcousticMetrics.jl is set up to require signed commits when merging into the `main` branch.
So, what to do?
Check out the CompatHelper.jl PR locally, manually sign the commits, then submit a new PR with the freshly-signed commits using this procedure:

* First, check the CompatHelper.jl PR locally: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/checking-out-pull-requests-locally?tool=cli
* Next, manually sign the CompatHelper.jl commit using git rebase: https://superuser.com/questions/397149/can-you-gpg-sign-old-commits#
* Then push the branch with the newly signed commits to my fork, and merge
* Close the CompatHelper.jl PR :-/

