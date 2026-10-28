// Cognito "pre sign-up" trigger for the atomSphere user pool.
//
// The pool signs people in by email, which is an alias of a random username, so Cognito
// lets a second account sign up with an email that's already taken and only refuses it
// at confirmation, after it has emailed a code that can't be used. This refuses the
// sign-up straight away instead, when a verified account already has that email.
// Unverified ones (sign-ups never finished) don't count, so starting again still works.
//
// Attach: User pool -> Extensions (Lambda triggers) -> Sign-up -> Pre sign-up.
// IAM: cognito-idp:ListUsers on the user pool.

import { CognitoIdentityProviderClient, ListUsersCommand } from '@aws-sdk/client-cognito-identity-provider';

const cognito = new CognitoIdentityProviderClient({ region: process.env.AWS_REGION || 'us-east-1' });

// Shown to the person signing up (Cognito prefixes it with "PreSignUp failed with error").
export const EMAIL_TAKEN = 'An account with this email already exists.';

export const handler = async (event) => {
  const email = event.request?.userAttributes?.email?.trim();
  // Accounts made by an admin, or linked from another sign-in provider, are left alone.
  if (!email || event.triggerSource !== 'PreSignUp_SignUp') return event;

  const result = await cognito.send(new ListUsersCommand({
    UserPoolId: event.userPoolId,
    // ListUsers filters on one attribute; quotes or backslashes would break the filter.
    Filter: `email = "${email.replace(/["\\]/g, '')}"`,
    AttributesToGet: ['email_verified'],
    Limit: 60,
  }));
  const taken = (result.Users ?? []).some((user) => (
    user.Username !== event.userName
    && user.Attributes?.some((attribute) => attribute.Name === 'email_verified' && attribute.Value === 'true')
  ));
  if (taken) throw new Error(EMAIL_TAKEN);
  return event;
};
