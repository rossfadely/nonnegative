import numpy as np

class PsfModels(object):
    """
    Create a set of psf patches, drawing
    from a power law in
    'score' = (flux)**2/(fwhm)**2/sigma_const**2 
    """
    def __init__(self,psfhwhm=1.5,
                 patchshape=(15,15),
                 npatches=10,
                 scorerange=(50,1000),
                 flux=1000,
                 bkg=0):
        self.psfhwhm = psfhwhm
        self.patchshape = patchshape
        self.npatches = npatches
        self.scorerange = scorerange
        self.flux = flux
        self.bkg = bkg

        assert ((np.mod(patchshape[0],2)) &
                (np.mod(patchshape[1],2))), \
            'Patch size must be odd by odd'
        
        # PSF stars in center of patch
        self.psfpos = ((patchshape[0]-1)/2,
            (patchshape[1]-1)/2)

        self.make_patches()
        self.avg_patches()

    def gaussianpsf(self,flux,xgrid,ygrid,x0,y0,psfhwhm):
        """
        Return a gaussian psf on grid of pixels
        """
        psf = flux * np.exp(-0.5 * ((xgrid-x0) ** 2 + (ygrid-y0) ** 2)
                            / psfhwhm ** 2) / np.sqrt(2. * np.pi * psfhwhm ** 2)
        return psf

    def make_patch(self,bkg_sigma):
        """
        Make a patch of pixels with constant noise and gaussian psf
        """
        patch = np.random.normal(size=self.patchshape) * bkg_sigma
        patch += self.bkg
        nx,ny = self.patchshape
        x,y   = np.meshgrid(range(nx),range(ny))
        patch += self.gaussianpsf(self.flux,x,y,self.psfpos[0],
                                  self.psfpos[1],self.psfhwhm)
        return patch

    def draw_score(self):
        """
        Draw score from a power law... how
        did I define that again? -2?
        """
        scoremin = self.scorerange[0]
        scoremax = self.scorerange[1]
        scorerng = scoremax-scoremin
        noscore = True
        while noscore:
            score = np.random.rand() * \
                (scorerng) + scoremin
            cdf   = scoremax * (score-scoremin) / \
                score / (scorerng)
            if 1-cdf > np.random.rand():
                  noscore = False
        return score

    def make_patches(self):
        """
        Make the patches
        """
        self.patches = np.zeros((self.npatches,self.patchshape[0],
                                 self.patchshape[1]))
        self.scores = np.zeros(self.npatches)
        self.bkg_sigmas = np.zeros(self.npatches)

        for i in range(self.npatches):
            self.scores[i] = self.draw_score()
            self.bkg_sigmas[i] = np.sqrt(self.flux**2. / (2*self.psfhwhm)**2. / \
                                         self.scores[i])
            self.patches[i] = self.make_patch(self.bkg_sigmas[i])

    def avg_patches(self):
        """
        Create an average psf which sums to one
        """
        self.avgpsf = np.zeros(self.patchshape)
        for i in range(self.npatches):
            self.avgpsf += self.patches[i] / self.patches[i].sum() / self.npatches
